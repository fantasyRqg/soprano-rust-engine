// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SopranoEngine",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "SopranoEngine", targets: ["SopranoEngine"]),
    ],
    targets: [
        .binaryTarget(
            name: "SopranoFFI",
            path: "SopranoFFI.xcframework"
        ),
        .target(
            name: "SopranoEngine",
            dependencies: ["SopranoFFI"],
            path: "Sources/SopranoEngine"
        ),
    ]
)
