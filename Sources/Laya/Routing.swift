import Foundation

public enum Checkpoint: String, Sendable, Codable, CaseIterable {
    case english, multilingual
    case typedDecisions = "typed-decisions"
    public init(name: String) throws {
        let key = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let aliases: [String: Self] = ["en": .english, "laya": .english, "default": .english,
            "multi": .multilingual, "ml": .multilingual, "laya-multilingual": .multilingual,
            "typed": .typedDecisions, "typed_decisions": .typedDecisions,
            "laya-typed-decisions": .typedDecisions, "decisions": .typedDecisions]
        guard let value = Self(rawValue: key) ?? aliases[key] else { throw LayaError.invalid("Unknown checkpoint: \(name)") }
        self = value
    }
    public var repository: String {
        "convaiinnovations/laya" + (self == .english ? "" : "/\(rawValue)")
    }
}

public struct LanguageDetection: Sendable, Codable, Equatable {
    public let script: String
    public let scriptProfile: [String: Double]
    public let language: String?
    public let isEnglish: Bool
    public let nonLatinFraction: Double
}

public enum LanguageDetector {
    private static let ranges: [(String, [ClosedRange<UInt32>])] = [
        ("greek", [0x0370...0x03FF, 0x1F00...0x1FFF]),
        ("cyrillic", [0x0400...0x052F, 0x2DE0...0x2DFF, 0xA640...0xA69F]),
        ("hebrew", [0x0590...0x05FF]),
        ("arabic", [0x0600...0x06FF, 0x0750...0x077F, 0x08A0...0x08FF, 0xFB50...0xFDFF, 0xFE70...0xFEFF]),
        ("devanagari", [0x0900...0x097F, 0xA8E0...0xA8FF]), ("bengali", [0x0980...0x09FF]),
        ("gurmukhi", [0x0A00...0x0A7F]), ("gujarati", [0x0A80...0x0AFF]), ("oriya", [0x0B00...0x0B7F]),
        ("tamil", [0x0B80...0x0BFF]), ("telugu", [0x0C00...0x0C7F]), ("kannada", [0x0C80...0x0CFF]),
        ("malayalam", [0x0D00...0x0D7F]), ("sinhala", [0x0D80...0x0DFF]), ("thai", [0x0E00...0x0E7F]),
        ("lao", [0x0E80...0x0EFF]), ("tibetan", [0x0F00...0x0FFF]), ("myanmar", [0x1000...0x109F]),
        ("georgian", [0x10A0...0x10FF]), ("ethiopic", [0x1200...0x137F]), ("khmer", [0x1780...0x17FF]),
        ("hangul", [0x1100...0x11FF, 0x3130...0x318F, 0xAC00...0xD7AF]),
        ("kana", [0x3040...0x309F, 0x30A0...0x30FF, 0x31F0...0x31FF]),
        ("han", [0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF]),
    ]
    private static let stopWords: [(String, Set<String>)] = [
        ("en", "the and is are was were to of in for with that this it you have has not but on at be as from will can would there their what which please we i"),
        ("fr", "le la les des une est pour dans que qui avec sur pas plus nous vous être cette mais sont ont aux ce"),
        ("de", "der die das und ist ein eine den dem nicht mit für auf von zu sich auch werden wurde haben sind oder aber"),
        ("es", "el los las que por con para una es se del como pero son está este esta todo más muy hay sus"),
        ("pt", "os as que em um uma para com não é se do da dos das mas são está este esta muito pelo pela"),
        ("it", "il lo gli che di per con non è si del della sono questo questa anche come più nella alla"),
        ("nl", "het een van is op te dat niet met voor zijn aan door maar ook worden deze naar wordt"),
    ].map { ($0.0, Set($0.1.split(separator: " ").map(String.init))) }
    private static let diacritics = Set("àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğı".unicodeScalars)

    private static func isLetter(_ s: Unicode.Scalar) -> Bool {
        switch s.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: true
        default: false
        }
    }
    /// Only string leaves, never English schema keys. Bound depth and emitted Unicode scalars.
    public static func stateText(_ state: JSONValue, maxCharacters: Int = 4000) -> String {
        guard maxCharacters > 0 else { return "" }
        var scalars = String.UnicodeScalarView()
        var haveLeaf = false
        func visit(_ value: JSONValue, _ depth: Int) {
            guard depth <= 6, scalars.count < maxCharacters else { return }
            switch value {
            case .string(let text):
                if haveLeaf { scalars.append(" ") }
                haveLeaf = true
                scalars.append(contentsOf: text.unicodeScalars.prefix(max(0, maxCharacters - scalars.count)))
            case .array(let values):
                for v in values { visit(v, depth + 1); if scalars.count >= maxCharacters { break } }
            case .object(let values):
                for key in values.keys.sorted() { visit(values[key]!, depth + 1); if scalars.count >= maxCharacters { break } }
            default: break
            }
        }
        visit(state, 0)
        return String(scalars)
    }
    public static func analyse(_ state: JSONValue) -> LanguageDetection {
        let text = stateText(state)
        var counts: [String: Int] = [:]
        var nonLatinOrder: [String] = []
        for s in text.unicodeScalars where isLetter(s) {
            let cp = s.value
            let script: String
            if cp < 0x0250 || (0x1E00...0x1EFF).contains(cp) { script = "latin" }
            else { script = ranges.first(where: { $0.1.contains(where: { $0.contains(cp) }) })?.0 ?? "other" }
            if script != "latin", counts[script] == nil { nonLatinOrder.append(script) }
            counts[script, default: 0] += 1
        }
        let total = counts.values.reduce(0, +)
        guard total > 0 else { return LanguageDetection(script: "unknown", scriptProfile: [:], language: nil, isEnglish: true, nonLatinFraction: 0) }
        let profile = counts.mapValues { Double($0) / Double(total) }
        // Match upstream's non-Latin-before-Latin tie handling, without Dictionary iteration dependence.
        let order = nonLatinOrder + ["latin"]
        var dominant = order[0]
        for key in order.dropFirst() where counts[key, default: 0] > counts[dominant, default: 0] { dominant = key }
        let language = dominant == "latin" ? guessLatinLanguage(text) : nil
        return LanguageDetection(script: dominant, scriptProfile: profile, language: language,
            isEnglish: dominant == "latin" && (language == nil || language == "en"),
            nonLatinFraction: DecisionMath.rounded(1 - (profile["latin"] ?? 0)))
    }
    public static func guessLatinLanguage(_ text: String) -> String? {
        let lowered = text.lowercased()
        let words = lowered.unicodeScalars.split(whereSeparator: { !isLetter($0) }).map { String(String.UnicodeScalarView($0)) }
        guard words.count >= 4 else { return nil }
        let scores = stopWords.map { (language, vocabulary) in (language, words.filter { vocabulary.contains($0) }.count) }
        let english = scores[0].1
        var best = scores[1]
        for score in scores.dropFirst(2) where score.1 > best.1 { best = score }
        let rate = Double(lowered.unicodeScalars.filter { diacritics.contains($0) }.count) / Double(max(1, lowered.unicodeScalars.count))
        if best.1 == 0 && rate < 0.02 { return english > 0 ? "en" : nil }
        if best.1 >= max(2, english + 2) || (rate >= 0.04 && best.1 >= english) { return best.0 }
        return english > 0 ? "en" : nil
    }
}

public struct RouteDecision: Sendable, Codable, Equatable {
    public let model: Checkpoint
    public let repo: String
    public let reason: String
    public let detection: LanguageDetection?
    public let workflow: String?
}

public struct RoutingPolicy: Sendable {
    public var defaultModel: Checkpoint
    public var autoTaskDetection: Bool
    public init(defaultModel: Checkpoint = .english, autoTaskDetection: Bool = false) {
        self.defaultModel = defaultModel; self.autoTaskDetection = autoTaskDetection
    }
    public static func workflow(questionIDs: Set<String>) -> String? {
        let workflows: [(String, Set<String>)] = [
            ("agent_trace_observability", ["action", "needs_review", "outcome", "risk", "urgency"]),
            ("customer_service", ["action", "category", "churn_risk", "needs_human", "urgency"]),
            ("invoice_processing", ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"]),
            ("security_incidents", ["credential_compromise", "disposition", "severity", "true_positive", "urgency"]),
        ]
        return workflows.first { $0.1 == questionIDs }?.0
    }
    /// Precedence: model > task > opt-in workflow > language > script/language heuristic > default.
    public func route(state: JSONValue, questionIDs: Set<String> = [], model: String? = nil,
                      task: String? = nil, language: String? = nil) throws -> RouteDecision {
        func result(_ key: Checkpoint, _ reason: String, _ detection: LanguageDetection? = nil, _ workflow: String? = nil) -> RouteDecision {
            RouteDecision(model: key, repo: key.repository, reason: reason, detection: detection, workflow: workflow)
        }
        if let model { return result(try Checkpoint(name: model), "explicit model=\(model)") }
        if let task { return result(try Checkpoint(name: task), "explicit task=\(task)") }
        let workflow = Self.workflow(questionIDs: questionIDs)
        if let workflow, autoTaskDetection { return result(.typedDecisions, "question ids match \(workflow)", nil, workflow) }
        if let language {
            let root = language.lowercased().replacingOccurrences(of: "_", with: "-").split(separator: "-").first.map(String.init) ?? ""
            guard !root.isEmpty else { throw LayaError.invalid("Empty language override.") }
            return result(["en", "eng", "english"].contains(root) ? .english : .multilingual, "explicit lang=\(language)", nil, workflow)
        }
        let d = LanguageDetector.analyse(state)
        if d.script == "unknown" { return result(defaultModel, "no letters detected; using default", d, workflow) }
        if d.script != "latin" { return result(.multilingual, "non-Latin script: \(d.script)", d, workflow) }
        if !d.isEnglish { return result(.multilingual, "Latin script; language looks like \(d.language ?? "unknown")", d, workflow) }
        return result(.english, "English Latin text", d, workflow)
    }
}

/// Model loads are single-flight; unloading invalidates late arrivals. The LRU limit bounds
/// cached agents, not agents still retained by an in-flight prediction or by the application.
public actor Router {
    public typealias Loader = @Sendable (Checkpoint) async throws -> Agent
    public nonisolated let policy: RoutingPolicy
    private let loader: Loader
    private struct Pending { let id: UUID; let task: Task<Agent, any Error> }
    private var agents: [Checkpoint: Agent] = [:]
    private var pending: [Checkpoint: Pending] = [:]
    private var revisions: [Checkpoint: UUID] = [:]
    private var order: [Checkpoint] = []
    public private(set) var maxLoaded: Int
    public var loaded: [Checkpoint] { order }

    public init(maxLoaded: Int = 1, policy: RoutingPolicy = .init(), loader: @escaping Loader) throws {
        guard maxLoaded >= 1 else { throw LayaError.invalid("maxLoaded must be positive.") }
        self.maxLoaded = maxLoaded; self.policy = policy; self.loader = loader
    }
    private func touch(_ key: Checkpoint) { order.removeAll { $0 == key }; order.append(key) }
    private func evict() {
        while order.count > maxLoaded { agents.removeValue(forKey: order.removeFirst()) }
    }
    public func load(_ key: Checkpoint) async throws -> Agent {
        try Task.checkCancellation()
        if let agent = agents[key] { touch(key); return agent }
        if revisions[key] == nil { revisions[key] = UUID() }
        let revision = revisions[key]!
        let item: Pending
        if let existing = pending[key] { item = existing }
        else {
            let loader = loader
            item = Pending(id: UUID(), task: Task { try await loader(key) })
            pending[key] = item
        }
        do {
            let agent = try await item.task.value
            guard revisions[key] == revision else { throw LayaError.loadInvalidated }
            if pending[key]?.id == item.id {
                pending.removeValue(forKey: key); agents[key] = agent; touch(key); evict()
            }
            try Task.checkCancellation()
            return agent
        } catch {
            if pending[key]?.id == item.id { pending.removeValue(forKey: key) }
            throw error
        }
    }
    @discardableResult public func attach(_ key: Checkpoint, agent: Agent) -> Agent {
        pending.removeValue(forKey: key)?.task.cancel(); revisions[key] = UUID()
        agents[key] = agent; touch(key); maxLoaded = max(maxLoaded, agents.count)
        return agent
    }
    public func preload(_ names: [Checkpoint] = Checkpoint.allCases) async throws {
        // Reserve the UNION, not max(existing.count, names.count), or an attached model is evicted.
        maxLoaded = max(maxLoaded, Set(agents.keys).union(names).count)
        for name in names { _ = try await load(name) }
    }
    public func unload(_ name: Checkpoint? = nil) {
        for key in name.map({ [$0] }) ?? Checkpoint.allCases {
            revisions[key] = UUID(); pending.removeValue(forKey: key)?.task.cancel()
            agents.removeValue(forKey: key); order.removeAll { $0 == key }
        }
    }
    public nonisolated func route(state: JSONValue, questionIDs: Set<String> = [], model: String? = nil,
                                   task: String? = nil, language: String? = nil) throws -> RouteDecision {
        try policy.route(state: state, questionIDs: questionIDs, model: model, task: task, language: language)
    }
    public func predict(state: JSONValue, questions: [Question], model: String? = nil,
                        task: String? = nil, language: String? = nil) async throws -> Prediction {
        let decision = try route(state: state, questionIDs: Set(questions.map(\.id)), model: model, task: task, language: language)
        let agent = try await load(decision.model)
        var result = try await agent.predict(state: state, questions: questions)
        result.routing = decision
        return result
    }
}
