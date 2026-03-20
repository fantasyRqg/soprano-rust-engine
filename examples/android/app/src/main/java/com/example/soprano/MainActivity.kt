package com.example.soprano

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import uniffi.soprano_ffi.ExecutionProvider

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val defaultModelPath = filesDir.resolve("soprano-models").absolutePath

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    TtsScreen(defaultModelPath = defaultModelPath)
                }
            }
        }
    }
}

@Composable
fun TtsScreen(
    defaultModelPath: String,
    viewModel: TtsViewModel = viewModel(),
) {
    val state by viewModel.uiState.collectAsState()
    var modelPath by rememberSaveable { mutableStateOf(defaultModelPath) }
    var text by rememberSaveable { mutableStateOf("Hello, this is a test of the Soprano text to speech engine.") }
    var selectedEp by rememberSaveable { mutableStateOf("CPU") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
    ) {
        Text("Soprano TTS Demo", style = MaterialTheme.typography.headlineMedium)

        Spacer(modifier = Modifier.height(16.dp))

        // Model path
        OutlinedTextField(
            value = modelPath,
            onValueChange = { modelPath = it },
            label = { Text("Model path") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Execution provider selector
        Text("Execution Provider:", style = MaterialTheme.typography.labelMedium)
        Spacer(modifier = Modifier.height(4.dp))
        Row {
            for (ep in listOf("CPU", "NNAPI", "XNNPACK")) {
                FilterChip(
                    selected = selectedEp == ep,
                    onClick = { selectedEp = ep },
                    label = { Text(ep) },
                    enabled = !state.isLoading && !state.isSpeaking,
                    modifier = Modifier.padding(end = 8.dp),
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Load button
        Row(verticalAlignment = Alignment.CenterVertically) {
            Button(
                onClick = {
                    val ep = when (selectedEp) {
                        "NNAPI" -> ExecutionProvider.NNAPI
                        "XNNPACK" -> ExecutionProvider.XNNPACK
                        else -> ExecutionProvider.CPU
                    }
                    viewModel.loadModel(modelPath, ep)
                },
                enabled = !state.isLoading && !state.isSpeaking,
            ) {
                Text("Load Model")
            }
            if (state.isLoading) {
                Spacer(modifier = Modifier.width(12.dp))
                CircularProgressIndicator(strokeWidth = 2.dp)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Text input
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("Text to speak") },
            minLines = 3,
            maxLines = 6,
            modifier = Modifier.fillMaxWidth(),
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Speak / Stop button
        Row {
            Button(
                onClick = { viewModel.speak(text) },
                enabled = state.isModelLoaded && !state.isSpeaking && text.isNotBlank(),
            ) {
                Text("Speak")
            }
            Spacer(modifier = Modifier.width(8.dp))
            Button(
                onClick = { viewModel.stop() },
                enabled = state.isSpeaking,
            ) {
                Text("Stop")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Status
        Text("Status: ${state.status}", style = MaterialTheme.typography.bodyMedium)

        // Metrics — each line as separate Text for accessibility/testability
        if (state.metrics.isNotEmpty()) {
            Spacer(modifier = Modifier.height(12.dp))
            Text("Metrics:", style = MaterialTheme.typography.titleSmall)
            Spacer(modifier = Modifier.height(4.dp))
            for (line in state.metrics.lines()) {
                if (line.isNotBlank()) {
                    Text(text = line, style = MaterialTheme.typography.bodySmall)
                }
            }
        }

        state.error?.let { error ->
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = error,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}
