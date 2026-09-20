import Foundation

/// Heuristic cleanup, not a security filter. Disable cleaning when quoted text matters.
public enum Email {
    public static func cleanBody(_ body: String, maxCharacters: Int = 3000) throws -> String {
        guard maxCharacters >= 0 else { throw LayaError.invalid("Negative email character limit.") }
        let quotes = [#"^\s*On .{0,300}wrote:\s*$"#, #"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}"#, #"^\s*_{8,}\s*$"#, #"^\s*From:\s.+$"#]
        let signatures = [#"^\s*--\s*$"#, #"^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$"#, #"^\s*sent from my (iphone|android|mobile|ipad)"#]
        func matches(_ text: String, _ patterns: [String]) -> Bool {
            patterns.contains { text.range(of: $0, options: [.regularExpression, .caseInsensitive]) != nil }
        }
        let text = body.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n").replacingOccurrences(of: "\\n", with: "\n")
        var lines: [String] = []
        for line in text.components(separatedBy: "\n") {
            if !lines.isEmpty && matches(line, quotes) { break }
            if line.trimmingCharacters(in: .whitespaces).hasPrefix(">") { continue }
            lines.append(line.replacingOccurrences(of: #"\s+$"#, with: "", options: .regularExpression))
        }
        let start = max(1, min(Int(Double(lines.count) * 0.6), lines.count - 8))
        if start < lines.count {
            for i in start..<lines.count {
                if lines[i].trimmingCharacters(in: .whitespaces).unicodeScalars.count <= 40 && matches(lines[i], signatures) {
                    lines = Array(lines.prefix(i)); break
                }
            }
        }
        let joined = lines.joined(separator: "\n")
        let separator = try NSRegularExpression(pattern: #"\n\s*\n"#)
        var paragraphs: [String] = [], position = joined.startIndex
        for match in separator.matches(in: joined, range: NSRange(joined.startIndex..., in: joined)) {
            guard let range = Range(match.range, in: joined) else { continue }
            paragraphs.append(String(joined[position..<range.lowerBound])); position = range.upperBound
        }
        paragraphs.append(String(joined[position...]))
        let disclaimer = #"(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|if you (have )?received this (e-?mail|message) in error)"#
        let cleaned = paragraphs.filter { !matches($0, [disclaimer]) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }.joined(separator: "\n\n")
            .replacingOccurrences(of: #"[ \t]+"#, with: " ", options: .regularExpression)
        return String(String.UnicodeScalarView(cleaned.unicodeScalars.prefix(maxCharacters)))
    }
    public static func state(subject: String, body: String, sender: String? = nil,
                             clean: Bool = true, extra: [String: JSONValue] = [:]) throws -> JSONValue {
        var fields: [String: JSONValue] = ["subject": .string(subject.trimmingCharacters(in: .whitespacesAndNewlines)),
                                           "body": .string(try clean ? cleanBody(body) : body)]
        if let sender, !sender.isEmpty { fields["from"] = .string(sender) }
        for (key, value) in extra where value != .null { fields[key] = value }
        return .object(fields)
    }
}
