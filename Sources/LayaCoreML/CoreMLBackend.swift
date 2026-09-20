import Foundation
import Laya

#if canImport(CoreML)
import CoreML

public enum ComputePolicy: Sendable { case cpuOnly, cpuAndGPU, all }

/// Owns Core ML reference types inside one actor. No MLModel or MLMultiArray crosses isolation.
/// The artifact must contain the complete encoder AND both decision heads, not just embeddings.
public actor CoreMLBackend: DecisionBackend {
    private let model: MLModel
    private let configuration: ModelConfiguration
    private let vocabularySize: Int
    private let actionCount: Int
    private let padTokenID: Int32
    private let sequenceLengths: [Int]
    private let modelOptionCount: Int
    private let temporaryCompiledURL: URL?

    public init(modelURL: URL, configuration: ModelConfiguration, vocabularySize: Int,
                actionCount: Int, padTokenID: Int32, compute: ComputePolicy = .cpuOnly) throws {
        try configuration.validate()
        guard modelURL.isFileURL, vocabularySize > 0, actionCount > 0, actionCount <= 256,
              padTokenID >= 0, Int(padTokenID) < vocabularySize else {
            throw LayaError.incompatibleModel("Invalid model URL, vocabulary size, or action count.")
        }
        let compiled: URL
        let temporary: URL?
        switch modelURL.pathExtension {
        case "mlmodelc": compiled = modelURL; temporary = nil
        case "mlpackage":
            compiled = try MLModel.compileModel(at: modelURL); temporary = compiled
        default: throw LayaError.incompatibleModel("Expected .mlpackage or .mlmodelc, not a PyTorch checkpoint.")
        }
        let settings = MLModelConfiguration()
        switch compute {
        case .cpuOnly: settings.computeUnits = .cpuOnly
        case .cpuAndGPU: settings.computeUnits = .cpuAndGPU
        case .all: settings.computeUnits = .all
        }
        do {
            let loaded = try MLModel(contentsOf: compiled, configuration: settings)
            let inputs = loaded.modelDescription.inputDescriptionsByName
            let names: Set<String> = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
            guard Set(inputs.keys) == names else {
                throw LayaError.incompatibleModel("Model input names do not match the SwiftLaya v1 tensor contract.")
            }
            for name in names {
                guard let description = inputs[name], description.type == .multiArray,
                      let constraint = description.multiArrayConstraint,
                      constraint.dataType == .int32,
                      constraint.shape.count == (name == "qtype" ? 1 : 2) else {
                    throw LayaError.incompatibleModel("Input \(name) must be an int32 tensor of the expected rank.")
                }
            }
            for name in ["logits", "act_logits"] {
                guard let output = loaded.modelDescription.outputDescriptionsByName[name], output.type == .multiArray else {
                    throw LayaError.incompatibleModel("Missing model output \(name).")
                }
            }
            guard let sequenceConstraint = inputs["input_ids"]?.multiArrayConstraint?.shapeConstraint,
                  let attentionConstraint = inputs["attention_mask"]?.multiArrayConstraint?.shapeConstraint else {
                throw LayaError.incompatibleModel("Missing sequence shape constraints.")
            }
            let sequenceShapes = sequenceConstraint.enumeratedShapes.map { $0.map(\.intValue) }
            let attentionShapes = attentionConstraint.enumeratedShapes.map { $0.map(\.intValue) }
            let lengths = sequenceShapes.compactMap { $0.count == 2 && $0[0] == 1 ? $0[1] : nil }.sorted()
            guard !lengths.isEmpty, sequenceShapes == attentionShapes,
                  let markerShape = inputs["marker_pos"]?.multiArrayConstraint?.shape.map(\.intValue),
                  inputs["marker_mask"]?.multiArrayConstraint?.shape.map(\.intValue) == markerShape,
                  markerShape.count == 2, markerShape[0] == 1, markerShape[1] >= 2,
                  inputs["qtype"]?.multiArrayConstraint?.shape.map(\.intValue) == [1] else {
                throw LayaError.incompatibleModel("Expected enumerated sequence lengths with fixed one-row marker tensors.")
            }
            self.model = loaded
            self.configuration = configuration
            self.vocabularySize = vocabularySize
            self.actionCount = actionCount
            self.padTokenID = padTokenID
            self.sequenceLengths = lengths
            self.modelOptionCount = markerShape[1]
            self.temporaryCompiledURL = temporary
        } catch {
            if let temporary { try? FileManager.default.removeItem(at: temporary) }
            throw error
        }
    }
    deinit {
        if let temporaryCompiledURL { try? FileManager.default.removeItem(at: temporaryCompiledURL) }
    }

    // Synchronous actor-isolated prediction satisfies the async protocol without yielding the MLModel.
    public func predict(_ batch: TokenBatch) throws -> ModelOutput {
        try Task.checkCancellation()
        try batch.validate()
        guard batch.batchSize <= configuration.maxQuestions,
              batch.sequenceLength <= configuration.maxLength,
              batch.optionCount <= configuration.maxOptions, batch.optionCount <= modelOptionCount,
              batch.inputIDs.allSatisfy({ Int($0) < vocabularySize }) else {
            throw LayaError.invalid("Input exceeds this exported model's limits or vocabulary.")
        }
        guard let paddedLength = sequenceLengths.first(where: { $0 >= batch.sequenceLength }) else {
            throw LayaError.invalid("No exported sequence length can hold this input.")
        }
        var logits: [[Double]] = []
        var actions: [[Double]] = []
        // ponytail: one Core ML call per question; add fixed-size batch artifacts only if profiling justifies them.
        for row in 0..<batch.batchSize {
            try Task.checkCancellation()
            var inputIDs = [Int32](repeating: padTokenID, count: paddedLength)
            var attentionMask = [Int32](repeating: 0, count: paddedLength)
            var markerPositions = [Int32](repeating: 0, count: modelOptionCount)
            var markerMask = [Int32](repeating: 0, count: modelOptionCount)
            let sequenceStart = row * batch.sequenceLength
            let optionStart = row * batch.optionCount
            inputIDs.replaceSubrange(0..<batch.sequenceLength,
                                     with: batch.inputIDs[sequenceStart..<(sequenceStart + batch.sequenceLength)])
            attentionMask.replaceSubrange(0..<batch.sequenceLength,
                                          with: batch.attentionMask[sequenceStart..<(sequenceStart + batch.sequenceLength)])
            markerPositions.replaceSubrange(0..<batch.optionCount,
                                            with: batch.markerPositions[optionStart..<(optionStart + batch.optionCount)])
            markerMask.replaceSubrange(0..<batch.optionCount,
                                       with: batch.markerMask[optionStart..<(optionStart + batch.optionCount)])
            let tensors: [String: MLMultiArray] = [
                "input_ids": try Self.tensor(inputIDs, shape: [1, paddedLength]),
                "attention_mask": try Self.tensor(attentionMask, shape: [1, paddedLength]),
                "marker_pos": try Self.tensor(markerPositions, shape: [1, modelOptionCount]),
                "marker_mask": try Self.tensor(markerMask, shape: [1, modelOptionCount]),
                "qtype": try Self.tensor([batch.questionTypes[row]], shape: [1]),
            ]
            let features = try MLDictionaryFeatureProvider(dictionary: tensors.mapValues { MLFeatureValue(multiArray: $0) })
            let prediction = try model.prediction(from: features)
            logits.append(Array(try Self.matrix(prediction, name: "logits", rows: 1, columns: modelOptionCount)[0].prefix(batch.optionCount)))
            actions.append(try Self.matrix(prediction, name: "act_logits", rows: 1, columns: actionCount)[0])
        }
        let result = ModelOutput(logits: logits, actionLogits: actions)
        try Task.checkCancellation()
        return result
    }
    private static func tensor(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        guard result.count == values.count else { throw LayaError.invalid("Tensor count mismatch.") }
        for (index, value) in values.enumerated() { result[index] = NSNumber(value: value) }
        return result
    }
    private static func matrix(_ features: any MLFeatureProvider, name: String, rows: Int, columns: Int) throws -> [[Double]] {
        guard let array = features.featureValue(for: name)?.multiArrayValue,
              array.shape.map({ $0.intValue }) == [rows, columns] else {
            throw LayaError.invalidOutput("Output \(name) has an unexpected shape.")
        }
        // Multi-index access respects strides, including non-contiguous Core ML outputs.
        return try (0..<rows).map { row in
            try (0..<columns).map { column in
                let value = array[[NSNumber(value: row), NSNumber(value: column)]].doubleValue
                guard value.isFinite else { throw LayaError.invalidOutput("Nonfinite \(name).") }
                return value
            }
        }
    }
}
#endif
