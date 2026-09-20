import Foundation

/// Versioned export metadata. Native code loads only local artifacts; it never downloads weights implicitly.
public struct ExportManifest: Sendable, Codable {
    public let formatVersion: Int
    public let coreMLVerified: Bool
    public let vocabularySize: Int
    public let actionCount: Int
    public let checkpoint: String
    public let sourceRevision: String
    public let sourceWeightSHA256: String
    public let absoluteTolerance: Double?
    public let relativeTolerance: Double?
    public func validate(allowUnverified: Bool = false) throws {
        guard formatVersion == 1, vocabularySize > 0, actionCount > 0,
              actionCount <= 256, sourceWeightSHA256.count == 64,
              sourceWeightSHA256.allSatisfy({ $0.isHexDigit }),
              absoluteTolerance.map({ $0.isFinite && $0 > 0 && $0 <= 0.01 }) ?? true,
              relativeTolerance.map({ $0.isFinite && $0 > 0 && $0 <= 0.0001 }) ?? true,
              coreMLVerified || allowUnverified else {
            throw LayaError.incompatibleModel("Invalid or unverified export. Run conversion verification on macOS first.")
        }
    }
}

/// Generated from the pinned Python reference, not from Swift's own implementation.
public struct ParityCase: Sendable, Codable {
    public let name: String
    public let request: PredictionRequest
    public let sequences: [TokenSequence]
    public let batch: TokenBatch
    public let output: ModelOutput
    public let answers: JSONValue
}

public enum Parity {
    public static func verifyTokens(_ fixture: ParityCase, tokenizer: any LayaTokenizer,
                                    configuration: ModelConfiguration) throws -> TokenBatch {
        guard fixture.request.questions.count == fixture.sequences.count else {
            throw LayaError.invalid("Malformed parity fixture.")
        }
        let sequences = try fixture.request.questions.map {
            try SequenceBuilder.build(state: fixture.request.state, question: $0, tokenizer: tokenizer, configuration: configuration)
        }
        for (a, b) in zip(sequences, fixture.sequences) {
            guard a.ids == b.ids, a.markers == b.markers, a.questionType == b.questionType else {
                throw LayaError.invalidOutput("Tokenizer/packing mismatch in \(fixture.name). Do not run this checkpoint in production.")
            }
        }
        let batch = try TokenBatch(sequences: sequences, padID: tokenizer.specialTokens.pad)
        try fixture.batch.validate()
        guard batch == fixture.batch else { throw LayaError.invalidOutput("Batch mismatch in \(fixture.name).") }
        return batch
    }
    public static func verifyOutput(_ actual: ModelOutput, against fixture: ParityCase,
                                    configuration: ModelConfiguration, tolerance: Double = 0.001,
                                    relativeTolerance: Double = 1e-6) throws {
        guard tolerance.isFinite, tolerance > 0,
              relativeTolerance.isFinite, relativeTolerance > 0 else {
            throw LayaError.invalid("Invalid parity tolerance.")
        }
        for (a, b) in [(actual.logits, fixture.output.logits), (actual.actionLogits, fixture.output.actionLogits)] {
            guard a.count == b.count else { throw LayaError.invalidOutput("Parity output row mismatch.") }
            for (ra, rb) in zip(a, b) {
                guard ra.count == rb.count else { throw LayaError.invalidOutput("Parity output column mismatch.") }
                for (x, y) in zip(ra, rb) {
                    let error = abs(x - y)
                    let relativeLimit = abs(y) * relativeTolerance
                    guard x.isFinite, y.isFinite, error <= tolerance + relativeLimit else {
                        throw LayaError.invalidOutput("Logit parity failed for \(fixture.name): \(x) vs \(y).")
                    }
                }
            }
        }
        let answers = try Postprocessor.process(actual, questions: fixture.request.questions, configuration: configuration)
        let encoded = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(answers))
        guard equal(encoded, fixture.answers, tolerance: tolerance) else {
            throw LayaError.invalidOutput("Answer parity failed for \(fixture.name).")
        }
    }
    private static func equal(_ a: JSONValue, _ b: JSONValue, tolerance: Double) -> Bool {
        func numeric(_ v: JSONValue) -> Double? {
            switch v { case .integer(let i): Double(i); case .number(let d): d; default: nil }
        }
        if let x = numeric(a), let y = numeric(b) { return x.isFinite && y.isFinite && abs(x - y) <= tolerance }
        switch (a, b) {
        case (.object(let x), .object(let y)):
            return Set(x.keys) == Set(y.keys) && x.allSatisfy { key, value in y[key].map { equal(value, $0, tolerance: tolerance) } ?? false }
        case (.array(let x), .array(let y)): return x.count == y.count && zip(x, y).allSatisfy { equal($0, $1, tolerance: tolerance) }
        default: return a == b
        }
    }
}
