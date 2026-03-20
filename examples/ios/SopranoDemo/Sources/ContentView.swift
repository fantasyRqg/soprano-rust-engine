import SwiftUI

struct ContentView: View {
    @State private var viewModel = TtsViewModel()
    @State private var text = "Hello, this is a test of the Soprano text to speech engine."
    @State private var selectedEp = "CPU"

    private var canLoad: Bool {
        !viewModel.isLoading && !viewModel.isSpeaking
    }

    private var canSpeak: Bool {
        viewModel.isModelLoaded && !viewModel.isSpeaking && !text.isEmpty
    }

    var body: some View {
        NavigationStack {
            List {
                // Engine section
                Section {
                    Picker("Execution Provider", selection: $selectedEp) {
                        Text("CPU").tag("CPU")
                    }
                    .disabled(!canLoad)

                    Button {
                        viewModel.loadModel()
                    } label: {
                        HStack {
                            Label("Load Model", systemImage: "cpu")
                            Spacer()
                            if viewModel.isLoading {
                                ProgressView()
                            } else if viewModel.isModelLoaded {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                            }
                        }
                    }
                    .disabled(!canLoad)
                } header: {
                    Text("Engine")
                }

                // Input section
                Section {
                    TextEditor(text: $text)
                        .frame(minHeight: 100)
                } header: {
                    Text("Text to Speak")
                }

                // Controls section
                Section {
                    HStack(spacing: 12) {
                        Button {
                            viewModel.speak(text: text)
                        } label: {
                            Label("Speak", systemImage: "play.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!canSpeak)

                        Button(role: .destructive) {
                            viewModel.stop()
                        } label: {
                            Label("Stop", systemImage: "stop.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .disabled(!viewModel.isSpeaking)
                    }
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                }

                // Status section
                Section {
                    LabeledContent("Status") {
                        HStack(spacing: 6) {
                            if viewModel.isSpeaking {
                                ProgressView()
                                    .controlSize(.small)
                            }
                            Text(viewModel.status)
                                .foregroundStyle(.secondary)
                        }
                    }

                    if !viewModel.metrics.isEmpty {
                        ForEach(viewModel.metrics.split(separator: "\n"), id: \.self) { line in
                            let parts = line.split(separator: ":", maxSplits: 1)
                            if parts.count == 2 {
                                LabeledContent(String(parts[0]).trimmingCharacters(in: .whitespaces)) {
                                    Text(parts[1].trimmingCharacters(in: .whitespaces))
                                        .fontDesign(.monospaced)
                                        .foregroundStyle(.secondary)
                                }
                            } else {
                                Text(line)
                                    .fontDesign(.monospaced)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                } header: {
                    Text("Status")
                }

                // Error section
                if let error = viewModel.error {
                    Section {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                            .font(.callout)
                    }
                }
            }
            .navigationTitle("Soprano TTS")
        }
        .onDisappear {
            viewModel.cleanup()
        }
    }
}
