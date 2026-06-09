@preconcurrency import AVFoundation
import SopranoEngine

final class AudioEngineSink: FfiAudioSink, @unchecked Sendable {

    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let playbackFormat: AVAudioFormat

    private let maxScheduledBuffers = 8
    private let semaphore = DispatchSemaphore(value: 8)
    // The engine writes one decode window at a time: 8 frames x 2048 samples
    // x 2 bytes = 32KB. 8 slots = ~256KB of scheduled audio.
    private let bufferSizePerChunk = 32_768

    // All mutable counters protected by this lock.
    // Accessed from: Rust callback thread (writePcm), AVAudioEngine completion queue,
    // main actor (reading metrics).
    private let lock = NSLock()
    private var _firstByteReceived = false
    private var _totalBytesWritten: Int64 = 0
    private var _bytesConsumed: Int64 = 0
    private var _totalBlockedNanos: Int64 = 0
    private var _buffersScheduled: Int = 0
    private var _buffersConsumed: Int = 0
    // Bumped by resetForNewSynthesis. Completion handlers of buffers from an
    // older generation must not signal the semaphore or touch counters: the
    // reset already restored all permits, so a late signal would push the
    // permit count above maxScheduledBuffers and quietly loosen backpressure.
    private var _generation: UInt64 = 0

    private let onStart: (_ tag: UInt64, _ sampleOffset: UInt64) -> Void
    private let onComplete: () -> Void
    private let onErr: (String) -> Void
    private let onFirstByte: () -> Void

    // Thread-safe accessors for metrics
    var totalBytesWritten: Int64 { lock.withLock { _totalBytesWritten } }
    var bytesConsumed: Int64 { lock.withLock { _bytesConsumed } }
    var totalBlockedNanos: Int64 { lock.withLock { _totalBlockedNanos } }
    var pendingBytes: Int64 { lock.withLock { _totalBytesWritten - _bytesConsumed } }

    init(
        onStart: @escaping (_ tag: UInt64, _ sampleOffset: UInt64) -> Void,
        onComplete: @escaping () -> Void,
        onErr: @escaping (String) -> Void,
        onFirstByte: @escaping () -> Void
    ) {
        self.onStart = onStart
        self.onComplete = onComplete
        self.onErr = onErr
        self.onFirstByte = onFirstByte

        // Use Float32 — Int16 is not supported on all audio outputs (e.g. iOS Simulator)
        self.playbackFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 32_000,
            channels: 1,
            interleaved: false
        )!

        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: playbackFormat)

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .spokenAudio)
            try session.setActive(true)
            try engine.start()
            playerNode.play()
        } catch {
            onErr("Audio engine setup failed: \(error.localizedDescription)")
        }
    }

    func writePcm(pcmData: Data) -> Int64 {
        let isFirst = lock.withLock {
            if !_firstByteReceived {
                _firstByteReceived = true
                return true
            }
            return false
        }
        if isFirst { onFirstByte() }

        // Convert raw Int16 LE bytes to Float32 AVAudioPCMBuffer
        let sampleCount = pcmData.count / 2
        guard sampleCount > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: playbackFormat, frameCapacity: AVAudioFrameCount(sampleCount))
        else {
            return -1
        }
        buffer.frameLength = AVAudioFrameCount(sampleCount)

        pcmData.withUnsafeBytes { rawPtr in
            guard let src = rawPtr.baseAddress?.assumingMemoryBound(to: Int16.self),
                  let dst = buffer.floatChannelData?[0] else { return }
            for i in 0..<sampleCount {
                dst[i] = Float(src[i]) / 32768.0
            }
        }

        // Backpressure: measure time blocked waiting for playback to catch up
        let waitStart = DispatchTime.now().uptimeNanoseconds
        semaphore.wait()
        let waitElapsed = DispatchTime.now().uptimeNanoseconds - waitStart

        let chunkBytes = Int64(pcmData.count)
        let generation = lock.withLock { _generation }
        playerNode.scheduleBuffer(buffer) { [weak self] in
            guard let self else { return }
            // Signal inside the lock so a reset can't interleave between the
            // generation check and the signal.
            self.lock.withLock {
                guard self._generation == generation else { return }
                self._buffersConsumed += 1
                self._bytesConsumed += chunkBytes
                self.semaphore.signal()
            }
        }

        lock.withLock {
            _totalBlockedNanos += Int64(waitElapsed)
            _buffersScheduled += 1
            _totalBytesWritten += chunkBytes
        }

        return Int64(pcmData.count)
    }

    func availableBytes() -> UInt64 {
        let (scheduled, consumed) = lock.withLock { (_buffersScheduled, _buffersConsumed) }
        let pending = scheduled - consumed
        let freeSlots = max(0, maxScheduledBuffers - pending)
        return UInt64(freeSlots * bufferSizePerChunk)
    }

    func onSentenceStart(tag: UInt64, sampleOffset: UInt64) {
        onStart(tag, sampleOffset)
    }

    func onDrainComplete() {
        onComplete()
    }

    func onError(tag: UInt64, message: String) {
        onErr("feed \(tag): \(message)")
    }

    /// Reset counters for a new synthesis. Bumping the generation first makes
    /// completion handlers of in-flight buffers no-ops: any handler that
    /// signalled before the bump did so under the lock, so its permit is
    /// already visible to the drain below; any handler after the bump sees a
    /// stale generation and stays silent.
    func resetForNewSynthesis() {
        lock.withLock {
            _generation &+= 1
            _firstByteReceived = false
            _totalBytesWritten = 0
            _bytesConsumed = 0
            _totalBlockedNanos = 0
            _buffersScheduled = 0
            _buffersConsumed = 0
        }
        playerNode.stop()
        // Drain whatever permits exist, then restore exactly maxScheduledBuffers.
        while case .success = semaphore.wait(timeout: .now()) {}
        for _ in 0..<maxScheduledBuffers { semaphore.signal() }
        playerNode.play()
    }

    func release() {
        playerNode.stop()
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false)
    }
}
