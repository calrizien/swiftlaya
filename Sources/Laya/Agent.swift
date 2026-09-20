import Foundation

public struct ModelOutput: Sendable, Codable, Equatable {
    public let logits: [[Double]]
    public let actionLogits: [[Double]]
    public init(logits: [[Double]], actionLogits: [[Double]]) { self.logits = logits; self.actionLogits = actionLogits }
}

/// Implementations must return unscaled option logits and unscaled action logits, not probabilities.
public protocol DecisionBackend: Sendable {
    func predict(_ batch: TokenBatch) async throws -> ModelOutput
}

public enum DecisionValue: Sendable, Equatable { case choice(String), score(Double), noul(Double) }

public struct Answer: Sendable, Equatable, Encodable {
    public let value: DecisionValue
    public let probabilities: [String: Double]
    public let confidence: Double
    public let actProbability: Double
    public let legend: [String: JSONValue]
    enum CodingKeys: String, CodingKey { case type, choice, score, noul, probabilities, confidence, action, legend }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch value {
        case .choice(let label):
            try c.encode("choice", forKey: .type); try c.encode(label, forKey: .choice)
            try c.encode(probabilities, forKey: .probabilities)
        case .score(let score):
            try c.encode("score", forKey: .type); try c.encode(score, forKey: .score)
            try c.encode(probabilities, forKey: .probabilities); try c.encode(legend, forKey: .legend)
        case .noul(let p): try c.encode("noul", forKey: .type); try c.encode(p, forKey: .noul)
        }
        try c.encode(confidence, forKey: .confidence)
        try c.encode(["act_probability": actProbability], forKey: .action)
    }
}

public struct Prediction: Sendable, Encodable {
    public let model: String
    public let answers: [String: Answer]
    public let usage: [String: Int]
    public let truncatedStateTokens: [String: Int]
    public var routing: RouteDecision?
}

public enum DecisionMath {
    public static func softmax(_ logits: [Double], temperature: Double = 1) throws -> [Double] {
        guard !logits.isEmpty, logits.allSatisfy(\.isFinite), temperature.isFinite, temperature > 0 else {
            throw LayaError.invalidOutput("Nonfinite logits, empty distribution, or invalid temperature.")
        }
        let scale = max(0.001, temperature)
        // Subtract first to avoid overflow when logits / temperature would overflow.
        let peak = logits.max()!
        let expValues = logits.map { exp(($0 - peak) / scale) }
        let total = expValues.reduce(0, +)
        return expValues.map { $0 / total }
    }
    public static func entropyConfidence(_ probabilities: [Double]) throws -> Double {
        guard probabilities.count >= 2, probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
              abs(probabilities.reduce(0, +) - 1) < 1e-6 else { throw LayaError.invalidOutput("Invalid probability distribution.") }
        let entropy = -probabilities.reduce(0) { $0 + $1 * log(max($1, 1e-12)) }
        return min(1, max(0, 1 - entropy / log(Double(probabilities.count))))
    }
    public static func temperatureBucket(type: QuestionType, options: Int) -> String {
        let size = options <= 2 ? "2" : options <= 5 ? "3-5" : options <= 10 ? "6-10" : "11+"
        return "\(type.rawValue):\(size)"
    }
    static func rounded(_ x: Double) -> Double { (x * 10_000).rounded(.toNearestOrEven) / 10_000 }
}

public enum Postprocessor {
    public static func process(_ output: ModelOutput, questions: [Question], configuration: ModelConfiguration) throws -> [String: Answer] {
        try configuration.validate()
        guard Set(questions.map(\.id)).count == questions.count,
              output.logits.count == questions.count, output.actionLogits.count == questions.count else {
            throw LayaError.invalidOutput("Duplicate question ids or wrong output batch size.")
        }
        var answers: [String: Answer] = [:]
        for (row, q) in questions.enumerated() {
            try q.validate(maxOptions: configuration.maxOptions)
            let k = q.labels.count
            guard output.logits[row].count >= k, !output.actionLogits[row].isEmpty else {
                throw LayaError.invalidOutput("Missing option or action logits for \(q.id).")
            }
            let t = configuration.temperatureByOptions[DecisionMath.temperatureBucket(type: q.type, options: k)]
                ?? configuration.temperature[Int(q.type.index)]
            let p = try DecisionMath.softmax(Array(output.logits[row].prefix(k)), temperature: t)
            let act = try DecisionMath.softmax(output.actionLogits[row])[0]
            let value: DecisionValue
            let confidence: Double
            var legend: [String: JSONValue] = [:]
            switch q.kind {
            case .choice:
                var top = 0
                for i in 1..<k where p[i] > p[top] { top = i }
                value = .choice(q.labels[top]); confidence = try DecisionMath.entropyConfidence(p)
            case .score(let levels):
                value = .score(DecisionMath.rounded(p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }))
                confidence = try DecisionMath.entropyConfidence(p)
                legend = Dictionary(uniqueKeysWithValues: levels.enumerated().map { (String($0.offset), $0.element) })
            case .noul:
                value = .noul(DecisionMath.rounded(p[1])); confidence = max(p[1], 1 - p[1])
            }
            answers[q.id] = Answer(value: value,
                probabilities: Dictionary(uniqueKeysWithValues: zip(q.labels, p.map(DecisionMath.rounded))),
                confidence: DecisionMath.rounded(confidence), actProbability: DecisionMath.rounded(act), legend: legend)
        }
        return answers
    }
}

/// Per-agent state is isolated; the backend owns its own inference synchronization.
public actor Agent {
    public nonisolated let configuration: ModelConfiguration
    private let tokenizer: any LayaTokenizer
    private let backend: any DecisionBackend
    public init(tokenizer: any LayaTokenizer, backend: any DecisionBackend, configuration: ModelConfiguration = .init()) throws {
        try configuration.validate()
        self.tokenizer = tokenizer; self.backend = backend; self.configuration = configuration
    }
    public func predict(state: JSONValue, questions: [Question]) async throws -> Prediction {
        try Task.checkCancellation()
        guard questions.count <= configuration.maxQuestions, Set(questions.map(\.id)).count == questions.count else {
            throw LayaError.invalid("Too many questions or duplicate question ids.")
        }
        if questions.isEmpty {
            return Prediction(model: "laya-rl-agent", answers: [:], usage: ["input_tokens": 0, "output_tokens": 0], truncatedStateTokens: [:])
        }
        let sequences = try questions.map { q in
            try Task.checkCancellation()
            return try SequenceBuilder.build(state: state, question: q, tokenizer: tokenizer, configuration: configuration)
        }
        let batch = try TokenBatch(sequences: sequences, padID: tokenizer.specialTokens.pad)
        try batch.validate()
        let output = try await backend.predict(batch)
        // Do not publish a result after an explicit cancellation, even if the backend cannot abort a kernel.
        try Task.checkCancellation()
        return Prediction(model: "laya-rl-agent",
            answers: try Postprocessor.process(output, questions: questions, configuration: configuration),
            usage: ["input_tokens": batch.inputTokenCount, "output_tokens": 0],
            truncatedStateTokens: Dictionary(uniqueKeysWithValues: zip(questions.map(\.id), sequences.map(\.truncatedStateTokens))))
    }
    public func systemOne(state: JSONValue, questions: [Question]) async throws -> Prediction {
        try await predict(state: state, questions: questions)
    }
}
