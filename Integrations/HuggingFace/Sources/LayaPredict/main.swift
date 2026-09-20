import Foundation
import Laya
import LayaHuggingFace
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

@main struct LayaPredict {
    static func main() async {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            guard args.count == 3, ["predict", "verify"].contains(args[0]) else {
                throw LayaError.invalid("Usage: laya-predict predict CHECKPOINT_DIR REQUEST.json\n       laya-predict verify CHECKPOINT_DIR parity.json")
            }
            #if canImport(CoreML)
            let checkpoint = try await LocalCheckpoint.load(from: URL(fileURLWithPath: args[1]))
            let input = try Data(contentsOf: URL(fileURLWithPath: args[2]))
            if args[0] == "verify" {
                let fixtures = try JSONDecoder().decode([ParityCase].self, from: input)
                try await checkpoint.verify(fixtures: fixtures)
                print("Verified \(fixtures.count) Python-to-Swift parity cases on CPU-only Core ML.")
            } else {
                let request = try JSONDecoder().decode(PredictionRequest.self, from: input)
                let result = try await checkpoint.agent.predict(state: request.state, questions: request.questions)
                let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
                FileHandle.standardOutput.write(try encoder.encode(result)); print("")
            }
            #else
            throw LayaError.incompatibleModel("Core ML inference requires an Apple platform. The Laya SDK and laya-route work on Linux.")
            #endif
        } catch {
            FileHandle.standardError.write(Data("laya-predict: \(error)\n".utf8)); exit(1)
        }
    }
}
