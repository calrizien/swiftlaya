import Foundation
import Laya

@main enum LayaRouteCommand {
    static func main() {
        do {
            var model: String?, language: String?, task: String?, input: String?
            var args = CommandLine.arguments.dropFirst().makeIterator()
            while let arg = args.next() {
                switch arg {
                case "--help", "-h":
                    print("Usage: laya-route [--model NAME] [--lang CODE] [--task NAME] [TEXT_OR_JSON]\nReads stdin when no text is supplied. Routing only: no checkpoint or inference required.")
                    return
                case "--model", "--lang", "--task":
                    guard let value = args.next() else { throw LayaError.invalid("Missing value for \(arg)") }
                    if arg == "--model" { model = value } else if arg == "--lang" { language = value } else { task = value }
                default:
                    guard !arg.hasPrefix("--"), input == nil else { throw LayaError.invalid("Unexpected argument: \(arg)") }
                    input = arg
                }
            }
            let text = input ?? String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
            let state = (try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8))) ?? .string(text)
            let route = try RoutingPolicy().route(state: state, model: model, task: task, language: language)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            print(String(decoding: try encoder.encode(route), as: UTF8.self))
        } catch {
            FileHandle.standardError.write(Data("laya-route: \(error)\n".utf8))
            exit(1)
        }
    }
}
