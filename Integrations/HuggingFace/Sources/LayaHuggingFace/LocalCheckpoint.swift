import Foundation
import Laya
import Tokenizers
#if canImport(CoreML)
import LayaCoreML
#endif

public struct HuggingFaceTokenizer: LayaTokenizer {
    private let tokenizer: any Tokenizer
    public let specialTokens: SpecialTokens

    /// Reads the exact tokenizer saved with the exported checkpoint. No Hub request is made.
    public static func load(from directory: URL, specialTokens: SpecialTokens) async throws -> Self {
        guard directory.isFileURL else { throw LayaError.invalid("Expected a local tokenizer directory.") }
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory, strict: true)
        guard tokenizer.convertTokenToId(specialTokens.maskToken) == Int(specialTokens.mask),
              [specialTokens.cls, specialTokens.sep, specialTokens.mask, specialTokens.pad].allSatisfy({ $0 >= 0 && tokenizer.convertIdToToken(Int($0)) != nil }),
              !specialTokens.maskToken.isEmpty else {
            throw LayaError.incompatibleModel("Special token metadata does not match this tokenizer.")
        }
        return Self(tokenizer: tokenizer, specialTokens: specialTokens)
    }
    public func encode(_ text: String) throws -> [Int32] {
        try tokenizer.encode(text: text, addSpecialTokens: false).map {
            guard let id = Int32(exactly: $0), id >= 0 else { throw LayaError.invalid("Token id outside int32 range.") }
            return id
        }
    }
}

#if canImport(CoreML)
/// Local-only native inference. Conversion is a separate, offline preparation step.
public struct LocalCheckpoint: Sendable {
    public let configuration: ModelConfiguration
    public let manifest: ExportManifest
    public let tokenizer: HuggingFaceTokenizer
    public let backend: CoreMLBackend
    public let agent: Agent

    public static func load(from directory: URL, compute: ComputePolicy = .cpuOnly,
                            allowUnverified: Bool = false) async throws -> Self {
        guard directory.isFileURL else { throw LayaError.invalid("Expected a local checkpoint directory.") }
        try Task.checkCancellation()
        let decoder = JSONDecoder()
        let configuration = try decoder.decode(ModelConfiguration.self, from: Data(contentsOf: directory.appendingPathComponent("rl_agent_config.json")))
        let manifest = try decoder.decode(ExportManifest.self, from: Data(contentsOf: directory.appendingPathComponent("swiftlaya.json")))
        try manifest.validate(allowUnverified: allowUnverified)
        let tokens = try decoder.decode(SpecialTokens.self, from: Data(contentsOf: directory.appendingPathComponent("special_tokens.json")))
        let tokenizer = try await HuggingFaceTokenizer.load(from: directory.appendingPathComponent("tokenizer"), specialTokens: tokens)
        let compiled = directory.appendingPathComponent("model.mlmodelc")
        let modelURL = FileManager.default.fileExists(atPath: compiled.path) ? compiled : directory.appendingPathComponent("model.mlpackage")
        // Keep compilation/loading off the caller's MainActor. No raw Core ML object is sent between tasks.
        let backend = try await Task.detached {
            try CoreMLBackend(modelURL: modelURL, configuration: configuration,
                              vocabularySize: manifest.vocabularySize, actionCount: manifest.actionCount, compute: compute)
        }.value
        try Task.checkCancellation()
        let agent = try Agent(tokenizer: tokenizer, backend: backend, configuration: configuration)
        return Self(configuration: configuration, manifest: manifest, tokenizer: tokenizer, backend: backend, agent: agent)
    }
    /// Exact token/mask parity; absolute-tolerance logit/probability parity; exact winning labels.
    public func verify(fixtures: [ParityCase], tolerance: Double = 0.001) async throws {
        guard !fixtures.isEmpty else { throw LayaError.invalid("Empty parity suite is not validation.") }
        for fixture in fixtures {
            try Task.checkCancellation()
            let batch = try Parity.verifyTokens(fixture, tokenizer: tokenizer, configuration: configuration)
            let output = try await backend.predict(batch)
            try Parity.verifyOutput(output, against: fixture, configuration: configuration, tolerance: tolerance)
        }
    }
}
#endif
