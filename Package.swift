// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "SwiftLaya",
    platforms: [.macOS(.v14), .iOS(.v17), .visionOS(.v1)],
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
