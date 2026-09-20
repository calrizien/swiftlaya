// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "SwiftLayaHuggingFace",
    platforms: [.macOS(.v15), .iOS(.v18), .visionOS(.v2)],
    products: [
        .library(name: "LayaHuggingFace", targets: ["LayaHuggingFace"]),
        .executable(name: "laya-predict", targets: ["LayaPredict"]),
    ],
    dependencies: [
        .package(name: "SwiftLaya", path: "../.."),
        .package(url: "https://github.com/huggingface/swift-transformers.git",
                 revision: "9088d55148b799e853cf4e039b0f0a1e3efe034c"),
    ],
    targets: [
        .target(name: "LayaHuggingFace", dependencies: [
            .product(name: "Laya", package: "SwiftLaya"),
            .product(name: "LayaCoreML", package: "SwiftLaya"),
            .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
        .executableTarget(name: "LayaPredict", dependencies: ["LayaHuggingFace", .product(name: "Laya", package: "SwiftLaya")]),
    ],
    swiftLanguageModes: [.v6]
)
