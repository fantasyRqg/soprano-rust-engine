import Foundation
import os
import SopranoEngine

private let logger = Logger(subsystem: "com.example.soprano-demo", category: "TTS")

/// Thread-safe timing state shared between main actor and background threads.
/// Write ordering is safe: feedStartNanos is set before engine.feed(),
/// firstByteMs is set during writePcm, and read only after engine.drain() completes.
/// Lock ensures formal correctness regardless of ordering assumptions.
final class TimingState: @unchecked Sendable {
    private let lock = NSLock()
    private var _feedStartNanos: UInt64 = 0
    private var _firstByteMs: Int64 = -1

    var feedStartNanos: UInt64 {
        get { lock.withLock { _feedStartNanos } }
        set { lock.withLock { _feedStartNanos = newValue } }
    }

    var firstByteMs: Int64 {
        get { lock.withLock { _firstByteMs } }
        set { lock.withLock { _firstByteMs = newValue } }
    }

    func reset() {
        lock.withLock {
            _feedStartNanos = 0
            _firstByteMs = -1
        }
    }
}

@Observable
@MainActor
final class TtsViewModel {

    var isModelLoaded = false
    var isLoading = false
    var isSpeaking = false
    var error: String?
    var status = "Tap Load to start"
    var metrics = ""

    private var engine: SopranoTts?
    private var sink: AudioEngineSink?
    private var speakTask: Task<Void, Never>?
    private let _timing = TimingState()

    func loadModel(ep: String = "CPU") {
        guard !isLoading else { return }
        isLoading = true
        error = nil
        status = "Loading model..."
        metrics = ""

        // Release previous engine/sink before creating new ones (#6)
        sink?.release()
        sink = nil
        engine = nil

        Task.detached { [weak self] in
            guard let self else { return }

            do {
                let modelPath = Bundle.main.resourcePath! + "/Models"

                let newSink = AudioEngineSink(
                    onSentence: { index in
                        Task { @MainActor in
                            self.status = "Sentence \(index + 1) complete"
                        }
                    },
                    onComplete: {
                        Task { @MainActor in
                            self.isSpeaking = false
                            self.status = "Done"
                        }
                    },
                    onErr: { msg in
                        logger.error("Sink error: \(msg)")
                        Task { @MainActor in
                            self.isSpeaking = false
                            self.error = msg
                            self.status = "Error"
                        }
                    },
                    onFirstByte: {
                        let elapsed = DispatchTime.now().uptimeNanoseconds - self._timing.feedStartNanos
                        self._timing.firstByteMs = Int64(elapsed / 1_000_000)
                    }
                )

                let config = SopranoConfig(
                    modelPath: modelPath,
                    temperature: 0.0,
                    topK: 0,
                    topP: 0.95,
                    repetitionPenalty: 1.2,
                    executionProvider: ep == "CoreML" ? .coreMl : .cpu
                )

                logger.info("Loading model from \(modelPath) with EP: \(ep)")
                let t0 = DispatchTime.now().uptimeNanoseconds
                let newEngine = try SopranoTts(config: config, sink: newSink)
                let loadMs = (DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000
                logger.info("Model loaded in \(loadMs)ms")

                await MainActor.run {
                    self.engine = newEngine
                    self.sink = newSink
                    self.isModelLoaded = true
                    self.isLoading = false
                    self.status = "Model loaded"
                    self.metrics = "Load: \(loadMs)ms"
                }
            } catch {
                logger.error("Model load failed: \(error.localizedDescription)")
                await MainActor.run {
                    self.isLoading = false
                    self.error = error.localizedDescription
                    self.status = "Failed to load model"
                }
            }
        }
    }

    func speak(text: String) {
        guard engine != nil, !isSpeaking else { return }
        isSpeaking = true
        error = nil
        status = "Synthesizing..."
        _timing.reset()
        sink?.resetForNewSynthesis()

        speakTask = Task.detached { [weak self] in
            guard let self, let engine = await self.engine else { return }

            do {
                self._timing.feedStartNanos = DispatchTime.now().uptimeNanoseconds
                let feedStart = DispatchTime.now().uptimeNanoseconds
                try engine.feed(text: text)
                engine.drain()
                let synthMs = (DispatchTime.now().uptimeNanoseconds - feedStart) / 1_000_000

                let totalBytes = await self.sink?.totalBytesWritten ?? 0
                let totalSamples = totalBytes / 2
                let audioDurationSec = Double(totalSamples) / 32_000.0

                let blockedMs = (await self.sink?.totalBlockedNanos ?? 0) / 1_000_000
                let inferenceMs = Int64(synthMs) - blockedMs
                let inferenceSec = Double(inferenceMs) / 1_000.0
                let rtf = inferenceSec > 0 ? audioDurationSec / inferenceSec : 0.0

                let firstByte = self._timing.firstByteMs

                await MainActor.run {
                    let loadLine = self.metrics.split(separator: "\n")
                        .first { $0.hasPrefix("Load:") }
                        .map(String.init) ?? ""

                    var lines: [String] = []
                    if !loadLine.isEmpty { lines.append(loadLine) }
                    if firstByte >= 0 { lines.append("First byte: \(firstByte)ms") }
                    lines.append("Inference: \(inferenceMs)ms (wall: \(synthMs)ms)")
                    if blockedMs > 0 {
                        lines.append("Audio blocked: \(blockedMs)ms")
                    }
                    lines.append(String(format: "Audio: %.2fs (%lld samples)", audioDurationSec, totalSamples))
                    lines.append(String(format: "RTF: %.2fx", rtf))
                    self.metrics = lines.joined(separator: "\n")
                    self.isSpeaking = false
                    self.status = "Done"
                    logger.info("Synthesis done — inference: \(inferenceMs)ms, RTF: \(String(format: "%.2f", rtf))x, first byte: \(firstByte)ms")
                }
            } catch {
                logger.error("Synthesis failed: \(error.localizedDescription)")
                await MainActor.run {
                    self.isSpeaking = false
                    self.error = error.localizedDescription
                    self.status = "Synthesis failed"
                }
            }
        }
    }

    func stop() {
        speakTask?.cancel()
        engine?.flush()
        // flush() causes the Rust side to return from blocked write,
        // which unblocks the semaphore naturally
        isSpeaking = false
        status = "Stopped"
    }

    func cleanup() {
        stop()
        engine = nil
        sink?.release()
        sink = nil
    }
}
