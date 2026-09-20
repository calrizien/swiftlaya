import Foundation

public struct SpecialTokens: Sendable, Codable, Equatable {
    public let cls: Int32
    public let sep: Int32
    public let mask: Int32
    public let pad: Int32
    public let maskToken: String
    public init(cls: Int32, sep: Int32, mask: Int32, pad: Int32, maskToken: String) throws {
        guard [cls, sep, mask, pad].allSatisfy({ $0 >= 0 }), !maskToken.isEmpty else {
            throw LayaError.invalid("Special token ids must be nonnegative and maskToken must be nonempty.")
        }
        self.cls = cls; self.sep = sep; self.mask = mask; self.pad = pad; self.maskToken = maskToken
    }
}

public protocol LayaTokenizer: Sendable {
    var specialTokens: SpecialTokens { get }
    /// Must encode without adding CLS/BOS/SEP/EOS. SequenceBuilder adds those explicitly.
    func encode(_ text: String) throws -> [Int32]
}

public struct ModelConfiguration: Sendable, Codable, Equatable {
    public var maxLength: Int
    public var headMaxLength: Int
    public var maxQuestions: Int
    public var maxOptions: Int
    public var temperature: [Double]
    public var temperatureByOptions: [String: Double]
    public init(maxLength: Int = 512, headMaxLength: Int = 192, maxQuestions: Int = 64,
                maxOptions: Int = 255, temperature: [Double] = [1, 1, 1],
                temperatureByOptions: [String: Double] = [:]) {
        self.maxLength = maxLength; self.headMaxLength = headMaxLength
        self.maxQuestions = maxQuestions; self.maxOptions = maxOptions
        self.temperature = temperature; self.temperatureByOptions = temperatureByOptions
    }
    public func validate() throws {
        guard (16...8192).contains(maxLength), (8..<maxLength).contains(headMaxLength),
              (1...256).contains(maxQuestions), (2...255).contains(maxOptions),
              temperature.count == 3, (temperature + Array(temperatureByOptions.values)).allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw LayaError.invalid("Invalid sequence limits or temperatures.")
        }
    }
    enum CodingKeys: String, CodingKey {
        case maxLength = "max_len", headMaxLength = "head_max_len", maxQuestions = "max_questions"
        case maxOptions = "max_options", temperature, temperatureByOptions = "temperature_by_options"
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(maxLength: try c.decodeIfPresent(Int.self, forKey: .maxLength) ?? 512,
                  headMaxLength: try c.decodeIfPresent(Int.self, forKey: .headMaxLength) ?? 192,
                  maxQuestions: try c.decodeIfPresent(Int.self, forKey: .maxQuestions) ?? 64,
                  maxOptions: try c.decodeIfPresent(Int.self, forKey: .maxOptions) ?? 255,
                  temperature: try c.decodeIfPresent([Double].self, forKey: .temperature) ?? [1, 1, 1],
                  temperatureByOptions: try c.decodeIfPresent([String: Double].self, forKey: .temperatureByOptions) ?? [:])
        try validate()
    }
}

public struct TokenSequence: Sendable, Codable, Equatable {
    public let ids: [Int32]
    public let markers: [Int32]
    public let questionType: Int32
    public let truncatedStateTokens: Int
    public init(ids: [Int32], markers: [Int32], questionType: Int32, truncatedStateTokens: Int = 0) {
        self.ids = ids; self.markers = markers; self.questionType = questionType
        self.truncatedStateTokens = truncatedStateTokens
    }
}

public enum SequenceBuilder {
    public static func build(state: JSONValue, question: Question, tokenizer: any LayaTokenizer,
                             configuration: ModelConfiguration = .init(),
                             truncateLeft: Bool = false, optionOrder: [Int]? = nil) throws -> TokenSequence {
        try configuration.validate(); try question.validate(maxOptions: configuration.maxOptions)
        let t = tokenizer.specialTokens
        guard !t.maskToken.isEmpty, [t.cls, t.sep, t.mask, t.pad].allSatisfy({ $0 >= 0 }) else {
            throw LayaError.invalid("Invalid special tokens.")
        }
        func clean(_ s: String) -> String { s.replacingOccurrences(of: t.maskToken, with: " ") }
        let options = try question.renderedOptions()
        let order = optionOrder ?? Array(options.indices)
        guard order.sorted() == Array(options.indices) else { throw LayaError.invalid("optionOrder must be a permutation.") }
        var head = try tokenizer.encode("\(question.type.rawValue) question: \(clean(question.instructions))")
        var optionIDs = try order.map { [t.mask] + Array(try tokenizer.encode(" " + clean(options[$0])).prefix(48)) }
        var budget = configuration.headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        if budget < 16 {
            let perOption = max(4, (configuration.headMaxLength - 16) / optionIDs.count)
            optionIDs = optionIDs.map { Array($0.prefix(perOption)) }
            budget = configuration.headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        }
        head = Array(head.prefix(max(8, budget)))
        var ids = [t.cls] + head + [t.sep]
        var markers: [Int32] = []
        for option in optionIDs { markers.append(Int32(ids.count)); ids += option }
        ids.append(t.sep)
        // Never silently remove labels or the final separator to fit an oversized head.
        guard ids.count < configuration.maxLength else { throw LayaError.invalid("Question \(question.id) options exceed the sequence budget.") }
        let stateIDs = try tokenizer.encode(clean(state.rendered()))
        let room = configuration.maxLength - ids.count - 1
        let kept = truncateLeft ? Array(stateIDs.suffix(room)) : Array(stateIDs.prefix(room))
        ids += kept; ids.append(t.sep)
        guard ids.allSatisfy({ $0 >= 0 }) else { throw LayaError.invalid("Tokenizer returned a negative token id.") }
        return TokenSequence(ids: ids, markers: markers, questionType: question.type.index,
                             truncatedStateTokens: stateIDs.count - kept.count)
    }
}

/// Flattened row-major arrays. Bounds are checked before allocating a batch.
public struct TokenBatch: Sendable, Codable, Equatable {
    public let batchSize: Int
    public let sequenceLength: Int
    public let optionCount: Int
    public let inputIDs: [Int32]
    public let attentionMask: [Int32]
    public let markerPositions: [Int32]
    public let markerMask: [Int32]
    public let questionTypes: [Int32]
    public var inputTokenCount: Int { attentionMask.reduce(0) { $0 + Int($1) } }

    public init(sequences: [TokenSequence], padID: Int32) throws {
        guard !sequences.isEmpty, sequences.count <= 256, padID >= 0 else { throw LayaError.invalid("Invalid batch size or padding token.") }
        for s in sequences {
            guard !s.ids.isEmpty, s.ids.count <= 8192, s.ids.allSatisfy({ $0 >= 0 }),
                  (2...255).contains(s.markers.count), Set(s.markers).count == s.markers.count,
                  s.markers.allSatisfy({ $0 >= 0 && Int($0) < s.ids.count }), (0...2).contains(s.questionType) else {
                throw LayaError.invalid("Malformed token sequence.")
            }
        }
        batchSize = sequences.count
        sequenceLength = sequences.map { $0.ids.count }.max()!
        optionCount = sequences.map { $0.markers.count }.max()!
        var ids = [Int32](repeating: padID, count: batchSize * sequenceLength)
        var attention = [Int32](repeating: 0, count: ids.count)
        var positions = [Int32](repeating: 0, count: batchSize * optionCount)
        var masks = [Int32](repeating: 0, count: positions.count)
        for (row, s) in sequences.enumerated() {
            for (i, id) in s.ids.enumerated() { ids[row * sequenceLength + i] = id; attention[row * sequenceLength + i] = 1 }
            for (i, pos) in s.markers.enumerated() { positions[row * optionCount + i] = pos; masks[row * optionCount + i] = 1 }
        }
        inputIDs = ids; attentionMask = attention; markerPositions = positions; markerMask = masks
        questionTypes = sequences.map(\.questionType)
    }

    /// Also validate decoded batches: synthesized Codable does not invoke the initializer above.
    public func validate() throws {
        guard (1...256).contains(batchSize), (1...8192).contains(sequenceLength), (2...255).contains(optionCount),
              inputIDs.count == batchSize * sequenceLength, attentionMask.count == inputIDs.count,
              markerPositions.count == batchSize * optionCount, markerMask.count == markerPositions.count,
              questionTypes.count == batchSize, inputIDs.allSatisfy({ $0 >= 0 }),
              questionTypes.allSatisfy({ (0...2).contains($0) }),
              attentionMask.allSatisfy({ $0 == 0 || $0 == 1 }), markerMask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
            throw LayaError.invalid("Malformed batch tensor shape or values.")
        }
        for row in 0..<batchSize {
            var live: [Int32] = []
            for col in 0..<optionCount {
                let idx = row * optionCount + col
                guard markerPositions[idx] >= 0, markerPositions[idx] < sequenceLength else { throw LayaError.invalid("Marker index out of range.") }
                if markerMask[idx] == 1 {
                    guard attentionMask[row * sequenceLength + Int(markerPositions[idx])] == 1 else { throw LayaError.invalid("Marker points into padding.") }
                    live.append(markerPositions[idx])
                }
            }
            guard live.count >= 2, Set(live).count == live.count else { throw LayaError.invalid("Missing or duplicate live markers.") }
        }
    }
}
