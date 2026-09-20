// Swift port of Laya. See NOTICE and the upstream Apache-2.0 LICENSE.
import Foundation

public enum LayaError: Error, Sendable, Equatable, CustomStringConvertible {
    case invalid(String)
    case incompatibleModel(String)
    case invalidOutput(String)
    case loadInvalidated

    public var description: String {
        switch self {
        case .invalid(let message), .incompatibleModel(let message), .invalidOutput(let message): message
        case .loadInvalidated: "The checkpoint load was invalidated by unload or replacement."
        }
    }
}

/// JSON without Any or unchecked Sendable. Object keys serialize in sorted order.
/// For byte-for-byte Python state serialization, pass that serialized JSON as .string.
public indirect enum JSONValue: Sendable, Equatable, Codable {
    case null, bool(Bool), integer(Int64), number(Double), string(String)
    case array([JSONValue]), object([String: JSONValue])

    public init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let x = try? c.decode(Bool.self) { self = .bool(x) }
        else if let x = try? c.decode(Int64.self) { self = .integer(x) }
        else if let x = try? c.decode(Double.self) { self = .number(x) }
        else if let x = try? c.decode(String.self) { self = .string(x) }
        else if let x = try? c.decode([JSONValue].self) { self = .array(x) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let x): try c.encode(x)
        case .integer(let x): try c.encode(x)
        case .number(let x): try c.encode(x)
        case .string(let x): try c.encode(x)
        case .array(let x): try c.encode(x)
        case .object(let x): try c.encode(x)
        }
    }

    /// Match Python's spacing and Unicode preservation; intentionally canonicalize object keys.
    public func jsonText() throws -> String {
        switch self {
        case .array(let a): return "[" + (try a.map { try $0.jsonText() }).joined(separator: ", ") + "]"
        case .object(let d):
            return "{" + (try d.keys.sorted().map {
                try JSONValue.string($0).jsonText() + ": " + d[$0]!.jsonText()
            }).joined(separator: ", ") + "}"
        default:
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.withoutEscapingSlashes]
            return String(decoding: try encoder.encode(self), as: UTF8.self)
        }
    }

    public func rendered() throws -> String {
        if case .string(let text) = self { return text }
        return try jsonText()
    }

    var isMissingDescription: Bool { self == .null || self == .string("") }
}

public enum QuestionType: String, Sendable, Codable, CaseIterable {
    case choice, score, noul
    public var index: Int32 {
        switch self { case .choice: 0; case .score: 1; case .noul: 2 }
    }
}

/// An array of options, not a Dictionary: label-to-logit order is part of the model input.
public struct ChoiceOption: Sendable, Equatable, Codable {
    public let label: String
    public let criterion: JSONValue
    public init(_ label: String, _ criterion: JSONValue = .null) {
        self.label = label
        self.criterion = criterion
    }
}

public struct Question: Sendable, Equatable, Codable {
    public enum Kind: Sendable, Equatable {
        case choice([ChoiceOption])
        case score([JSONValue])
        case noul(falseCriterion: JSONValue, trueCriterion: JSONValue)
    }
    public let id: String
    public let instructions: String
    public let kind: Kind
    public init(id: String, instructions: String, kind: Kind) {
        self.id = id; self.instructions = instructions; self.kind = kind
    }
    public static func noul(_ id: String, _ instructions: String,
                            falseCriterion: JSONValue = .null, trueCriterion: JSONValue = .null) -> Self {
        Self(id: id, instructions: instructions, kind: .noul(falseCriterion: falseCriterion, trueCriterion: trueCriterion))
    }
    public var type: QuestionType {
        switch kind { case .choice: .choice; case .score: .score; case .noul: .noul }
    }
    public var labels: [String] {
        switch kind {
        case .choice(let a): a.map(\.label)
        case .score(let a): a.indices.map(String.init)
        case .noul: ["false", "true"]
        }
    }
    public func validate(maxOptions: Int = 255) throws {
        guard (2...255).contains(maxOptions) else { throw LayaError.invalid("Invalid maximum option count.") }
        guard !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw LayaError.invalid("Empty question id.") }
        let labels = labels
        guard (2...maxOptions).contains(labels.count), Set(labels).count == labels.count,
              labels.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw LayaError.invalid("Question \(id) requires 2...\(maxOptions) distinct, nonempty options.")
        }
    }
    public func renderedOptions() throws -> [String] {
        switch kind {
        case .choice(let options):
            return try options.map { $0.criterion.isMissingDescription ? $0.label : try $0.label + ": " + $0.criterion.rendered() }
        case .score(let levels):
            return try levels.enumerated().map { try "level \($0.offset): " + $0.element.rendered() }
        case .noul(let f, let t):
            return ["false: " + (f.isMissingDescription ? "no, the statement does not hold" : try f.rendered()),
                    "true: " + (t.isMissingDescription ? "yes, the statement holds" : try t.rendered())]
        }
    }
    private enum CodingKeys: String, CodingKey { case id, type, instructions, criteria }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        instructions = try c.decode(String.self, forKey: .instructions)
        switch try c.decode(QuestionType.self, forKey: .type) {
        case .choice: kind = .choice(try c.decode([ChoiceOption].self, forKey: .criteria))
        case .score: kind = .score(try c.decode([JSONValue].self, forKey: .criteria))
        case .noul:
            let d = try c.decodeIfPresent([String: JSONValue].self, forKey: .criteria) ?? [:]
            guard Set(d.keys).isSubset(of: ["false", "true"]) else { throw LayaError.invalid("Unknown noul criterion.") }
            kind = .noul(falseCriterion: d["false"] ?? .null, trueCriterion: d["true"] ?? .null)
        }
        try validate()
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id); try c.encode(type, forKey: .type); try c.encode(instructions, forKey: .instructions)
        switch kind {
        case .choice(let a): try c.encode(a, forKey: .criteria)
        case .score(let a): try c.encode(a, forKey: .criteria)
        case .noul(let f, let t): try c.encode(["false": f, "true": t], forKey: .criteria)
        }
    }
}

public struct PredictionRequest: Sendable, Codable {
    public let state: JSONValue
    public let questions: [Question]
    public init(state: JSONValue, questions: [Question]) { self.state = state; self.questions = questions }
}
