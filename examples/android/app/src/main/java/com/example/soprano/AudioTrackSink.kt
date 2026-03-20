package com.example.soprano

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import uniffi.soprano_ffi.FfiAudioSink

class AudioTrackSink(
    private val onSentence: (UInt) -> Unit,
    private val onComplete: () -> Unit,
    private val onErr: (String) -> Unit,
    private val onFirstByte: () -> Unit,
) : FfiAudioSink {

    private val sampleRate = 32000
    private val bufferSizeBytes: Int
    private val audioTrack: AudioTrack
    @Volatile private var firstByteReceived = false
    @Volatile var totalBytesWritten: Long = 0
        private set
    @Volatile var totalWriteNanos: Long = 0
        private set

    init {
        val minBuffer = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        bufferSizeBytes = minBuffer * 4

        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setBufferSizeInBytes(bufferSizeBytes)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        audioTrack.play()
    }

    override fun writePcm(pcmData: ByteArray): Long {
        if (!firstByteReceived) {
            firstByteReceived = true
            onFirstByte()
        }
        val t0 = System.nanoTime()
        val written = audioTrack.write(pcmData, 0, pcmData.size)
        totalWriteNanos += System.nanoTime() - t0
        if (written > 0) totalBytesWritten += written
        return if (written < 0) -1L else written.toLong()
    }

    fun resetForNewSynthesis() {
        firstByteReceived = false
        totalBytesWritten = 0
        totalWriteNanos = 0
    }

    override fun availableBytes(): ULong {
        return bufferSizeBytes.toULong()
    }

    override fun onSentenceComplete(sentenceIndex: UInt) {
        onSentence(sentenceIndex)
    }

    override fun onDrainComplete() {
        onComplete()
    }

    override fun onError(message: String) {
        onErr(message)
    }

    fun release() {
        try {
            audioTrack.stop()
        } catch (_: IllegalStateException) {}
        audioTrack.release()
    }
}
