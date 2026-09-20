import Foundation
import Testing
@testable import Laya

// Deliberately synthetic tokenizer for SDK contract tests, never a production tokenizer.
private struct ScalarTokenizer: LayaTokenizer {
    let specialTokens = try! SpecialTokens(cls: 1, sep: 2, mask: 3, pad: 0, maskToken: "<M>")
    func encode(_ text: String) -> [Int32] { text.unicodeScalars.map { Int32($0.value) + 10 } }
}
private func choice(_ id: String = "c", _ labels: [String] = ["first", "second"]) -> Question {
    Question(id: id, instructions: "Choose", kind: .choice(labels.map { ChoiceOption($0) }))
}
private actor TestBackend: DecisionBackend {
    var calls = 0
    func predict(_ b: TokenBatch) throws -> ModelOutput {
        calls += 1
        try b.validate()
        return ModelOutput(logits: (0..<b.batchSize).map { _ in (0..<b.optionCount).map(Double.init) },
                           actionLogits: (0..<b.batchSize).map { _ in [0, 0] })
    }
}
private func makeAgent(backend: TestBackend = TestBackend()) throws -> Agent {
    try Agent(tokenizer: ScalarTokenizer(), backend: backend)
}
private actor CountingLoader {
    var calls: [Checkpoint: Int] = [:]
    func load(_ key: Checkpoint) throws -> Agent { calls[key, default: 0] += 1; return try makeAgent() }
}
private actor GatedLoader {
    private var result: CheckedContinuation<Agent, any Error>?
    private var observers: [CheckedContinuation<Void, Never>] = []
    private var started = false
    func load() async throws -> Agent {
        try await withCheckedThrowingContinuation { continuation in
            result = continuation; started = true
            observers.forEach { $0.resume() }; observers.removeAll()
        }
    }
    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { observers.append($0) }
    }
    func release() throws { result?.resume(returning: try makeAgent()); result = nil }
}

@Test func structuredCriteriaKeepJSONAndFalse() throws {
    let q = Question(id: "x", instructions: "x", kind: .choice([
        .init("zero", .integer(0)), .init("no", .bool(false)), .init("dict", .object(["desc": .string("münchen")])), .init("empty", .string(""))]))
    #expect(try q.renderedOptions() == ["zero: 0", "no: false", "dict: {\"desc\": \"münchen\"}", "empty"])
}
@Test func noulIsFalseThenTrue() throws {
    let q = Question.noul("n", "Is this phishing?", falseCriterion: .object(["desc": .string("legitimate")]), trueCriterion: .array([.integer(1)]))
    #expect(try q.renderedOptions() == ["false: {\"desc\": \"legitimate\"}", "true: [1]"])
}
@Test func defaultNoulDescriptions() throws {
    #expect(try Question.noul("n", "x").renderedOptions() == ["false: no, the statement does not hold", "true: yes, the statement holds"])
}
@Test func orderedScoreLevels() throws {
    let q = Question(id: "s", instructions: "x", kind: .score([.string("low"), .object(["d": .string("high")]), .integer(2)]))
    #expect(try q.renderedOptions() == ["level 0: low", "level 1: {\"d\": \"high\"}", "level 2: 2"])
}
@Test func canonicalJSONAndRawState() throws {
    let state = JSONValue.object(["z": .string("é/雪"), "a": .bool(false)])
    #expect(try state.rendered() == "{\"a\": false, \"z\": \"é/雪\"}")
    #expect(try JSONValue.string("{\"z\": 1}").rendered() == "{\"z\": 1}")
    let decoded = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(state))
    #expect(decoded == state)
}
@Test func rejectNonfiniteJSON() { #expect(throws: (any Error).self) { try JSONValue.number(.nan).jsonText() } }
@Test func schemasRoundTripWithoutReordering() throws {
    let q = choice("x", ["z", "a", "m"])
    #expect(try JSONDecoder().decode(Question.self, from: JSONEncoder().encode(q)) == q)
}
@Test(arguments: [[], ["one"], ["a", "a"], ["", "a"]]) func rejectsInvalidChoice(labels: [String]) {
    #expect(throws: LayaError.self) { try choice("x", labels).validate() }
}
@Test func rejectsMalformedConfiguration() {
    #expect(throws: LayaError.self) { try ModelConfiguration(temperature: [1, .nan, 1]).validate() }
    #expect(throws: LayaError.self) { try ModelConfiguration(maxLength: 12).validate() }
    #expect(throws: LayaError.self) { try ModelConfiguration(temperature: [1, -1, 1]).validate() }
}
@Test func configDecodesPythonKeys() throws {
    let c = try JSONDecoder().decode(ModelConfiguration.self, from: Data(#"{"encoder":"anything","max_len":1024,"head_max_len":256,"temperature_by_options":{"choice:2":2}}"#.utf8))
    #expect(c.maxLength == 1024); #expect(c.headMaxLength == 256); #expect(c.temperatureByOptions["choice:2"] == 2)
}
@Test func exactSequenceLayout() throws {
    let t = ScalarTokenizer()
    let q = Question(id: "x", instructions: "i", kind: .choice([.init("A"), .init("B")]))
    let s = try SequenceBuilder.build(state: .string("S"), question: q, tokenizer: t)
    var expected: [Int32] = [1]
    expected += t.encode("choice question: i")
    expected += [2, 3]
    expected += t.encode(" A")
    expected += [3]
    expected += t.encode(" B")
    expected += [2]
    expected += t.encode("S")
    expected += [2]
    #expect(s.ids == expected)
    #expect(s.markers == [20, 23]); #expect(s.questionType == 0)
}
@Test func maskInjectionIsRemoved() throws {
    let q = Question(id: "x", instructions: "<M>", kind: .choice([.init("<M>"), .init("B", .string("<M>"))]))
    let t = ScalarTokenizer()
    let s = try SequenceBuilder.build(state: .string("text <M>"), question: q, tokenizer: t)
    #expect(s.ids.filter { $0 == t.specialTokens.mask }.count == 2)
    #expect(!s.ids.contains(Int32(Unicode.Scalar("<").value) + 10))
}
@Test func optionPermutationValidated() throws {
    #expect(throws: LayaError.self) { try SequenceBuilder.build(state: .null, question: choice(), tokenizer: ScalarTokenizer(), optionOrder: [1, 1]) }
    let s = try SequenceBuilder.build(state: .null, question: choice(), tokenizer: ScalarTokenizer(), optionOrder: [1, 0])
    #expect(s.markers.count == 2)
}
@Test func preservesFinalSeparatorAndReportsTruncation() throws {
    let c = ModelConfiguration(maxLength: 48, headMaxLength: 24)
    let s = try SequenceBuilder.build(state: .string(String(repeating: "x", count: 500)), question: choice(), tokenizer: ScalarTokenizer(), configuration: c)
    #expect(s.ids.count == 48); #expect(s.ids.last == 2); #expect(s.truncatedStateTokens > 0)
}
@Test func zeroStateRoomLeftTruncationDoesNotKeepWholeState() throws {
    let q = Question(id: "x", instructions: "i", kind: .choice([.init("A"), .init("B")]))
    let blank = try SequenceBuilder.build(state: .string(""), question: q, tokenizer: ScalarTokenizer())
    let c = ModelConfiguration(maxLength: blank.ids.count, headMaxLength: 24)
    let full = try SequenceBuilder.build(state: .string("abcdefghijklmnopqrstuvwxyz"), question: q, tokenizer: ScalarTokenizer(), configuration: c, truncateLeft: true)
    #expect(full.ids == blank.ids); #expect(full.truncatedStateTokens == 26)
}
@Test func oversizedQuestionThrowsRatherThanDroppingOptions() {
    let q = choice("x", (0..<30).map(String.init))
    #expect(throws: LayaError.self) { try SequenceBuilder.build(state: .null, question: q, tokenizer: ScalarTokenizer(), configuration: .init(maxLength: 32, headMaxLength: 16)) }
}
@Test func raggedBatchMasksAndTokenUsage() throws {
    let a = TokenSequence(ids: [1, 3, 3, 2], markers: [1, 2], questionType: 2)
    let b = TokenSequence(ids: [1, 3, 3, 3, 2], markers: [1, 2, 3], questionType: 0)
    let batch = try TokenBatch(sequences: [a, b], padID: 0)
    try batch.validate()
    #expect(batch.inputIDs == [1, 3, 3, 2, 0, 1, 3, 3, 3, 2])
    #expect(batch.markerMask == [1, 1, 0, 1, 1, 1]); #expect(batch.inputTokenCount == 9)
}
@Test func malformedBatchIsRejected() {
    #expect(throws: LayaError.self) { try TokenBatch(sequences: [.init(ids: [1, 2], markers: [1, 3], questionType: 0)], padID: 0) }
}
@Test func stableSoftmax() throws {
    let p = try DecisionMath.softmax([10_000, 10_001])
    #expect(abs(p[1] - 0.7310585786) < 1e-9)
    #expect(try DecisionMath.softmax([Double.greatestFiniteMagnitude, Double.greatestFiniteMagnitude], temperature: 0.001) == [0.5, 0.5])
}
@Test func entropyConfidenceIsNotTopProbability() throws {
    #expect(abs(try DecisionMath.entropyConfidence([0.5, 0.5])) < 1e-12)
    #expect(abs(try DecisionMath.entropyConfidence([1, 0]) - 1) < 1e-12)
    #expect(throws: LayaError.self) { try DecisionMath.entropyConfidence([0.1, 0.1]) }
}
@Test func allThreePrimitivesAndPaddedLogits() throws {
    let qs = [choice(), Question(id: "s", instructions: "rate", kind: .score([.integer(0), .integer(1), .integer(2)])), Question.noul("n", "yes?")]
    let r = try Postprocessor.process(.init(logits: [[0, 0, 99_999], [0, 0, 0], [0, 0, 99_999]], actionLogits: [[0, 0], [0, 0], [0, 0]]), questions: qs, configuration: .init())
    #expect(r["c"]?.value == .choice("first")); #expect(r["c"]?.confidence == 0)
    #expect(r["s"]?.value == .score(1)); #expect(r["n"]?.value == .noul(0.5)); #expect(r["n"]?.confidence == 0.5)
    #expect(r["n"]?.actProbability == 0.5)
}
@Test func perOptionTemperatureOverridesTypeTemperature() throws {
    let config = ModelConfiguration(temperature: [1, 1, 1], temperatureByOptions: ["choice:2": 2])
    let r = try Postprocessor.process(.init(logits: [[0, 2]], actionLogits: [[0, 0]]), questions: [choice()], configuration: config)
    #expect(r["c"]?.probabilities["second"] == 0.7311)
}
@Test(arguments: [2, 3, 5, 6, 10, 11, 255]) func temperatureBuckets(k: Int) {
    let expected = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+"
    #expect(DecisionMath.temperatureBucket(type: .score, options: k) == "score:" + expected)
}
@Test func rejectsNonfiniteOutput() {
    #expect(throws: LayaError.self) { try Postprocessor.process(.init(logits: [[0, .nan]], actionLogits: [[0, 0]]), questions: [choice()], configuration: .init()) }
}
@Test func answerJSONMatchesPythonWireShape() throws {
    let r = try Postprocessor.process(.init(logits: [[0, 0]], actionLogits: [[0, 0]]), questions: [Question.noul("n", "x")], configuration: .init())
    let json = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(r["n"]))
    #expect(json == .object(["type": .string("noul"), "noul": .number(0.5), "confidence": .number(0.5), "action": .object(["act_probability": .number(0.5)])]))
}
@Test func agentUsesOneBatchAndEmptyIsNoOp() async throws {
    let backend = TestBackend(), agent = try makeAgent(backend: backend)
    let empty = try await agent.predict(state: .null, questions: [])
    #expect(empty.answers.isEmpty); #expect(await backend.calls == 0)
    let result = try await agent.predict(state: .string("text"), questions: [choice(), Question.noul("n", "yes?")])
    #expect(result.answers.count == 2); #expect(await backend.calls == 1); #expect(result.usage["output_tokens"] == 0)
}
@Test func duplicateQuestionIDsDoNotReachBackend() async throws {
    let backend = TestBackend(), agent = try makeAgent(backend: backend)
    await #expect(throws: LayaError.self) { try await agent.predict(state: .null, questions: [choice(), choice()]) }
    #expect(await backend.calls == 0)
}
@Test(arguments: [("मुझे मदद चाहिए", "devanagari"), ("你好世界", "han"), ("안녕하세요", "hangul"), ("Привет мир", "cyrillic"), ("مرحبا", "arabic"), ("שלום", "hebrew"), ("γειά", "greek")])
func detectsNonLatin(text: String, script: String) {
    let d = LanguageDetector.analyse(.string(text))
    #expect(d.script == script); #expect(!d.isEnglish)
}
@Test func languageDetectionIgnoresKeys() throws {
    let state = JSONValue.object(["the and is for with that this please": .string("你好")])
    #expect(try RoutingPolicy().route(state: state).model == .multilingual)
}
@Test func germanStopWordsRouteMultilingual() throws {
    #expect(try RoutingPolicy().route(state: .string("Der Kunde wurde zweimal belastet und das ist nicht korrekt")).model == .multilingual)
}
@Test func englishAndNoText() throws {
    #expect(try RoutingPolicy().route(state: .string("Please help with the bill")).model == .english)
    #expect(try RoutingPolicy(defaultModel: .multilingual).route(state: .integer(42)).model == .multilingual)
}
@Test func routingOverridePrecedenceAndAliases() throws {
    let policy = RoutingPolicy(autoTaskDetection: true)
    let ids: Set<String> = ["action", "category", "churn_risk", "needs_human", "urgency"]
    #expect(try policy.route(state: .string("你好"), questionIDs: ids, model: " en ", task: "typed", language: "de").model == .english)
    #expect(try policy.route(state: .null, task: "typed_decisions", language: "en").model == .typedDecisions)
    #expect(try policy.route(state: .null, questionIDs: ids, language: "en").model == .typedDecisions)
    #expect(try RoutingPolicy().route(state: .null, questionIDs: ids, language: "en-US").model == .english)
    #expect(try RoutingPolicy().route(state: .null, language: "EN_us").model == .english)
    #expect(throws: LayaError.self) { try Checkpoint(name: "garbage") }
}
@Test func workflowMatchMustBeExactAndOptIn() throws {
    let ids: Set<String> = ["action", "category", "churn_risk", "needs_human", "urgency"]
    #expect(RoutingPolicy.workflow(questionIDs: ids) == "customer_service")
    #expect(RoutingPolicy.workflow(questionIDs: ids.union(["other"])) == nil)
    #expect(try RoutingPolicy().route(state: .string("hello"), questionIDs: ids).model == .english)
}
@Test func stateTextIsBounded() {
    #expect(LanguageDetector.stateText(.string(String(repeating: "a", count: 10_000))).count == 4000)
    #expect(LanguageDetector.stateText(.null) == "")
}
@Test func routerLRUEviction() async throws {
    let loads = CountingLoader()
    let router = try Router { try await loads.load($0) }
    _ = try await router.load(.english); _ = try await router.load(.multilingual)
    #expect(await router.loaded == [.multilingual])
    _ = try await router.load(.english)
    #expect(await loads.calls[.english] == 2)
}
@Test func preloadReservesUnionIncludingAttachedAgent() async throws {
    let loads = CountingLoader()
    let router = try Router { try await loads.load($0) }
    _ = await router.attach(.english, agent: try makeAgent())
    try await router.preload([.multilingual, .typedDecisions])
    #expect(await router.loaded.count == 3); #expect(await router.maxLoaded == 3)
    #expect(await loads.calls[.english] == nil)
}
@Test func concurrentLoadsAreSingleFlight() async throws {
    let loads = CountingLoader()
    let router = try Router { try await loads.load($0) }
    try await withThrowingTaskGroup(of: Void.self) { group in
        for _ in 0..<20 { group.addTask { _ = try await router.load(.english) } }
        try await group.waitForAll()
    }
    #expect(await loads.calls[.english] == 1)
}
@Test func unloadedSlowLoadCannotResurrectCheckpoint() async throws {
    let gate = GatedLoader(), router = try Router { _ in try await gate.load() }
    let task = Task { try await router.load(.english) }
    await gate.waitUntilStarted(); await router.unload(.english); try await gate.release()
    await #expect(throws: LayaError.loadInvalidated) { try await task.value }
    #expect(await router.loaded.isEmpty)
}
@Test func cancellingWaiterDoesNotPublishItsResult() async throws {
    let gate = GatedLoader(), router = try Router { _ in try await gate.load() }
    let task = Task { try await router.load(.english) }
    await gate.waitUntilStarted(); task.cancel(); try await gate.release()
    await #expect(throws: CancellationError.self) { try await task.value }
    // The shared load may still populate the cache for non-cancelled callers.
    #expect(await router.loaded == [.english])
}

@Test func presetsAreValidAndIndependent() throws {
    for questions in [Presets.triage(), Presets.email(), Presets.guardrails(), Presets.moderation(), Presets.router()] {
        #expect(Set(questions.map(\.id)).count == questions.count)
        for q in questions { try q.validate(); #expect(try q.renderedOptions().count == q.labels.count) }
    }
    #expect(Presets.triage().count == 5); #expect(Presets.router().count == 4)
    #expect(Presets.email(categories: [.init("z"), .init("a")])[0].labels == ["z", "a"])
}
@Test func emailStripsQuoteAndSignature() throws {
    let body = "Please refund me.\r\n\r\nThanks,\r\nPerson\r\nOn Monday Bob wrote:\r\nOld text"
    #expect(try Email.cleanBody(body) == "Please refund me.")
    #expect(try Email.cleanBody("new text\n> old quote\ncontinued") == "new text\ncontinued")
}
@Test func emailCleanupIsOptionalAndBounded() throws {
    #expect(try Email.state(subject: " x ", body: "confidential", clean: false) == .object(["subject": .string("x"), "body": .string("confidential")]))
    #expect(try Email.cleanBody("abcdef", maxCharacters: 3) == "abc")
    #expect(try Email.cleanBody("abcdef", maxCharacters: 0) == "")
    #expect(throws: LayaError.self) { try Email.cleanBody("x", maxCharacters: -1) }
}
@Test func emailExtrasMatchReferenceOverrideSemantics() throws {
    let state = try Email.state(subject: "old", body: "body", sender: "sender", extra: ["subject": .string("new"), "none": .null])
    #expect(state == .object(["subject": .string("new"), "body": .string("body"), "from": .string("sender")]))
}
@Test func invalidOptionLimitDoesNotTrap() {
    #expect(throws: LayaError.self) { try choice().validate(maxOptions: -1) }
}
@Test func realRequestExampleDecodes() throws {
    let request = #"{"state":{"message":"hello"},"questions":[{"id":"c","type":"choice","instructions":"x","criteria":[{"label":"z","criterion":null},{"label":"a","criterion":false}]}]}"#
    let decoded = try JSONDecoder().decode(PredictionRequest.self, from: Data(request.utf8))
    #expect(decoded.questions[0].labels == ["z", "a"])
}
@Test func unverifiedExportsFailClosed() throws {
    let text = #"{"formatVersion":1,"coreMLVerified":false,"vocabularySize":50000,"actionCount":2,"checkpoint":"local","sourceRevision":"local","sourceWeightSHA256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","absoluteTolerance":0.00001,"relativeTolerance":0.000001}"#
    let manifest = try JSONDecoder().decode(ExportManifest.self, from: Data(text.utf8))
    #expect(manifest.absoluteTolerance == 0.00001)
    #expect(manifest.relativeTolerance == 0.000001)
    #expect(throws: LayaError.self) { try manifest.validate() }
    try manifest.validate(allowUnverified: true)
}
@Test func parityRejectsWrongTokensAndOutputs() throws {
    let q = choice(), t = ScalarTokenizer(), c = ModelConfiguration()
    let sequence = try SequenceBuilder.build(state: .string("x"), question: q, tokenizer: t)
    let batch = try TokenBatch(sequences: [sequence], padID: 0)
    let output = ModelOutput(logits: [[0, 1]], actionLogits: [[0, 0]])
    let answers = try Postprocessor.process(output, questions: [q], configuration: c)
    let encoded = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(answers))
    let fixture = ParityCase(name: "contract", request: .init(state: .string("x"), questions: [q]), sequences: [sequence], batch: batch, output: output, answers: encoded)
    #expect(try Parity.verifyTokens(fixture, tokenizer: t, configuration: c) == batch)
    try Parity.verifyOutput(output, against: fixture, configuration: c)
    #expect(throws: LayaError.self) { try Parity.verifyOutput(.init(logits: [[0, 2]], actionLogits: [[0, 0]]), against: fixture, configuration: c) }
    let large = ModelOutput(logits: [[0, 1]], actionLogits: [[4389.991211, -3591.664551]])
    let largeAnswers = try Postprocessor.process(large, questions: [q], configuration: c)
    let largeEncoded = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(largeAnswers))
    let largeFixture = ParityCase(name: "large-logits", request: fixture.request, sequences: [sequence], batch: batch,
                                  output: large, answers: largeEncoded)
    try Parity.verifyOutput(.init(logits: [[0, 1]], actionLogits: [[4389.9883, -3591.662]]),
                            against: largeFixture, configuration: c)
    #expect(throws: LayaError.self) {
        try Parity.verifyOutput(.init(logits: [[0, 1]], actionLogits: [[4389.9883, -3591.662]]),
                                against: largeFixture, configuration: c,
                                tolerance: 0.001, relativeTolerance: 1e-12)
    }
    let wrong = ParityCase(name: "bad", request: .init(state: .string("changed"), questions: [q]), sequences: [sequence], batch: batch, output: output, answers: encoded)
    #expect(throws: LayaError.self) { try Parity.verifyTokens(wrong, tokenizer: t, configuration: c) }
}
