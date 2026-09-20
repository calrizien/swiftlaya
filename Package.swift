// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "SwiftLaya",
    platforms: [.macOS(.v15), .iOS(.v18), .visionOS(.v2)],
    products: [
        .library(name: "Laya", targets: ["Laya"]),
        .library(name: "LayaCoreML", targets: ["LayaCoreML"]),
        .executable(name: "laya-route", targets: ["LayaCLI"]),
    ],
    targets: [
        .target(name: "Laya"),
        .target(name: "LayaCoreML", dependencies: ["Laya"]),
        .executableTarget(name: "LayaCLI", dependencies: ["Laya"]),
        .testTarget(name: "LayaTests", dependencies: ["Laya"]),
    ],
    swiftLanguageModes: [.v6]
)
