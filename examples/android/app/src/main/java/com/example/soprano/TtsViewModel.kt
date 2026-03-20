package com.example.soprano

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import uniffi.soprano_ffi.ExecutionProvider
import uniffi.soprano_ffi.SopranoConfig
import uniffi.soprano_ffi.SopranoTts

data class TtsUiState(
    val isModelLoaded: Boolean = false,
    val isLoading: Boolean = false,
    val isSpeaking: Boolean = false,
    val error: String? = null,
    val status: String = "Enter model path and tap Load",
    val metrics: String = "",
)

class TtsViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(TtsUiState())
    val uiState: StateFlow<TtsUiState> = _uiState

    private var engine: SopranoTts? = null
    private var sink: AudioTrackSink? = null
    private var speakJob: Job? = null

    @Volatile private var feedStartNanos: Long = 0
    @Volatile private var firstByteMs: Long = -1

    fun loadModel(modelPath: String, executionProvider: ExecutionProvider = ExecutionProvider.CPU) {
        if (_uiState.value.isLoading) return

        _uiState.update { it.copy(isLoading = true, error = null, status = "Loading model...", metrics = "") }

        viewModelScope.launch(Dispatchers.IO) {
            try {
                engine?.close()
                sink?.release()

                val newSink = AudioTrackSink(
                    onSentence = { index ->
                        _uiState.update { it.copy(status = "Sentence ${index + 1u} complete") }
                    },
                    onComplete = {
                        _uiState.update { it.copy(isSpeaking = false, status = "Done") }
                    },
                    onErr = { msg ->
                        Log.e("SopranoTts", "Engine error: $msg")
                        _uiState.update { it.copy(isSpeaking = false, error = msg, status = "Error") }
                    },
                    onFirstByte = {
                        firstByteMs = (System.nanoTime() - feedStartNanos) / 1_000_000
                    },
                )

                val config = SopranoConfig(
                    modelPath = modelPath,
                    temperature = 0.0f,
                    topK = 0u,
                    topP = 0.95f,
                    repetitionPenalty = 1.2f,
                    executionProvider = executionProvider,
                )

                val t0 = System.nanoTime()
                val newEngine = SopranoTts(config, newSink)
                val loadMs = (System.nanoTime() - t0) / 1_000_000

                engine = newEngine
                sink = newSink

                _uiState.update {
                    it.copy(
                        isModelLoaded = true,
                        isLoading = false,
                        status = "Model loaded",
                        metrics = "Load: ${loadMs}ms",
                    )
                }
            } catch (e: Exception) {
                Log.e("SopranoTts", "Failed to load model", e)
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        error = e.message ?: "Unknown error",
                        status = "Failed to load model",
                    )
                }
            }
        }
    }

    fun speak(text: String) {
        val eng = engine ?: return
        if (_uiState.value.isSpeaking) return

        _uiState.update { it.copy(isSpeaking = true, error = null, status = "Synthesizing...") }

        // Reset tracking
        firstByteMs = -1
        sink?.resetForNewSynthesis()

        speakJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                feedStartNanos = System.nanoTime()
                eng.feed(text)
                eng.drain()
                val synthMs = (System.nanoTime() - feedStartNanos) / 1_000_000

                // Actual audio duration from bytes written: bytes / 2 (16-bit) / 32000 Hz
                val totalBytes = sink?.totalBytesWritten ?: 0L
                val totalSamples = totalBytes / 2
                val audioDurationSec = totalSamples.toDouble() / 32000.0

                // Subtract time blocked in AudioTrack.write() to get true inference time
                val writeBlockMs = (sink?.totalWriteNanos ?: 0L) / 1_000_000
                val inferenceMs = synthMs - writeBlockMs
                val inferenceSec = inferenceMs.toDouble() / 1000.0
                val rtf = if (inferenceSec > 0) audioDurationSec / inferenceSec else 0.0

                val loadPart = _uiState.value.metrics
                    .lineSequence()
                    .firstOrNull { it.startsWith("Load:") }
                    ?: ""

                _uiState.update {
                    it.copy(
                        metrics = buildString {
                            if (loadPart.isNotEmpty()) appendLine(loadPart)
                            if (firstByteMs >= 0) appendLine("First byte: ${firstByteMs}ms")
                            appendLine("Inference: ${inferenceMs}ms (wall: ${synthMs}ms)")
                            appendLine("Audio: %.2fs (%d samples)".format(audioDurationSec, totalSamples))
                            append("RTF: %.2fx".format(rtf))
                        },
                    )
                }
            } catch (e: Exception) {
                Log.e("SopranoTts", "Synthesis failed", e)
                _uiState.update {
                    it.copy(isSpeaking = false, error = e.message, status = "Synthesis failed")
                }
            }
        }
    }

    fun stop() {
        engine?.flush()
        speakJob?.cancel()
        _uiState.update { it.copy(isSpeaking = false, status = "Stopped") }
    }

    override fun onCleared() {
        super.onCleared()
        engine?.close()
        sink?.release()
    }
}
