import Constants from "expo-constants";
import { StatusBar } from "expo-status-bar";
import * as ImagePicker from "expo-image-picker";
import React, { useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  KeyboardAvoidingView,
  Linking,
  NativeModules,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

type Detection = {
  label: string;
  confidence: number;
};

type DetectionRow = {
  label: string;
  confidence: number;
  count: number;
};

type BestGuess = {
  label: string;
  confidence: number;
  count: number;
} | null;

type Diagnostics = {
  image_size?: string;
  boxes_at_selected_conf?: number;
  boxes_at_probe_conf?: number | null;
  max_conf_at_probe?: number | null;
  adaptive_conf_used?: number | null;
  best_guess_label?: string | null;
  best_guess_confidence?: number | null;
  best_guess_support?: number | null;
} | null;

type PredictionResponse = {
  ok: boolean;
  modelFilename: string;
  modelPath: string;
  modelClasses: string[];
  conf: number;
  imgsz: number;
  imageSize: {
    width: number;
    height: number;
  };
  resultImage: string;
  detections: Detection[];
  detectionRows: DetectionRow[];
  diagnostics: Diagnostics;
  bestGuess: BestGuess;
  detectionHint: string | null;
  error?: string;
};

type HealthResponse = {
  ok: boolean;
  modelFilename: string;
  modelPath: string;
  modelClasses: string[];
  defaultConf: number;
  imgsz: number;
};

type SelectedImage = {
  uri: string;
  fileName: string;
  mimeType: string;
  width?: number;
  height?: number;
};

const DEFAULT_CONF_TEXT = "0.25";
const DEFAULT_SERVER_PORT = 5008;
const QUICK_CONF_VALUES = ["0.10", "0.25", "0.40", "0.60"];

function extractHost(value?: string | null): string | null {
  if (!value) {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  const urlMatch = trimmed.match(/^(?:exp|exps|http|https):\/\/([^/:?#]+)(?::\d+)?/i);
  if (urlMatch?.[1]) {
    return urlMatch[1];
  }

  const rawHostMatch = trimmed.match(/^([^/:?#\s]+)(?::\d+)?$/i);
  return rawHostMatch?.[1] ?? null;
}

function inferServerCandidates(): string[] {
  const sourceCode = (NativeModules as { SourceCode?: { scriptURL?: string } }).SourceCode;
  const manifest = Constants.manifest as
    | {
        debuggerHost?: string;
        hostUri?: string;
      }
    | null
    | undefined;
  const manifest2 = (Constants as unknown as {
    manifest2?: {
      extra?: {
        expoClient?: {
          hostUri?: string;
        };
      };
    };
  }).manifest2;

  const rawValues = [
    sourceCode?.scriptURL,
    Constants.linkingUri,
    manifest?.debuggerHost,
    manifest?.hostUri,
    manifest2?.extra?.expoClient?.hostUri,
  ];

  const uniqueHosts = Array.from(
    new Set(rawValues.map(extractHost).filter((value): value is string => Boolean(value)))
  );

  return uniqueHosts.map((host) => `http://${host}:${DEFAULT_SERVER_PORT}`);
}

function normalizeServerUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

async function fetchJsonWithTimeout(
  url: string,
  options?: RequestInit,
  timeoutMs = 8000
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

function clampConfidence(value: string): number {
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    return Number(DEFAULT_CONF_TEXT);
  }
  return Math.min(0.95, Math.max(0.01, parsed));
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function guessMimeType(uri: string, mimeType?: string | null): string {
  if (mimeType) {
    return mimeType;
  }

  const lower = uri.toLowerCase();
  if (lower.endsWith(".png")) {
    return "image/png";
  }
  if (lower.endsWith(".webp")) {
    return "image/webp";
  }
  return "image/jpeg";
}

function buildFileName(asset: ImagePicker.ImagePickerAsset): string {
  if (asset.fileName) {
    return asset.fileName;
  }

  const extension = guessMimeType(asset.uri, asset.mimeType).split("/")[1] ?? "jpg";
  return `orchid-${Date.now()}.${extension}`;
}

async function pickImage(
  source: "library" | "camera"
): Promise<SelectedImage | null> {
  if (source === "library") {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission needed", "Please allow photo access to choose an image.");
      return null;
    }
  } else {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission needed", "Please allow camera access to take a photo.");
      return null;
    }
  }

  const options: ImagePicker.ImagePickerOptions = {
    mediaTypes: ["images"],
    allowsEditing: true,
    quality: 1,
  };

  const result =
    source === "library"
      ? await ImagePicker.launchImageLibraryAsync(options)
      : await ImagePicker.launchCameraAsync(options);

  if (result.canceled || !result.assets.length) {
    return null;
  }

  const asset = result.assets[0];
  return {
    uri: asset.uri,
    fileName: buildFileName(asset),
    mimeType: guessMimeType(asset.uri, asset.mimeType),
    width: asset.width,
    height: asset.height,
  };
}

async function parseResponseJson(response: Response) {
  const rawText = await response.text();
  if (!rawText) {
    return null;
  }

  try {
    return JSON.parse(rawText);
  } catch {
    return { ok: false, error: rawText };
  }
}

export default function App() {
  const inferredServerCandidates = useMemo(() => inferServerCandidates(), []);
  const [serverUrl, setServerUrl] = useState<string>(() => inferredServerCandidates[0] ?? "");
  const [confidenceText, setConfidenceText] = useState(DEFAULT_CONF_TEXT);
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [serverMessage, setServerMessage] = useState(
    inferredServerCandidates[0]
      ? `Detected server candidate: ${inferredServerCandidates[0]}`
      : "Use the same Wi-Fi network as the laptop running the Flask server."
  );
  const [loading, setLoading] = useState(false);
  const [checkingServer, setCheckingServer] = useState(false);

  const normalizedServerUrl = useMemo(() => normalizeServerUrl(serverUrl), [serverUrl]);
  const serverCandidates = useMemo(
    () =>
      Array.from(
        new Set(
          [normalizedServerUrl, ...inferredServerCandidates]
            .map(normalizeServerUrl)
            .filter(Boolean)
        )
      ),
    [inferredServerCandidates, normalizedServerUrl]
  );
  const confidence = useMemo(() => clampConfidence(confidenceText), [confidenceText]);
  const previewImageUri = selectedImage?.uri ?? null;
  const annotatedImageUri = prediction ? `data:image/jpeg;base64,${prediction.resultImage}` : null;
  const topDetection = prediction?.detectionRows?.[0] ?? null;

  const handleSelectImage = async (source: "library" | "camera") => {
    const image = await pickImage(source);
    if (!image) {
      return;
    }

    setSelectedImage(image);
    setPrediction(null);
  };

  const handleTestServer = async () => {
    if (!serverCandidates.length) {
      Alert.alert("Server URL missing", "Enter your laptop server URL first.");
      return;
    }

    setCheckingServer(true);
    try {
      let lastError = "The mobile app could not reach the Flask server.";

      for (const candidate of serverCandidates) {
        try {
          const response = await fetchJsonWithTimeout(`${candidate}/api/health`);
          const data = (await parseResponseJson(response)) as HealthResponse | null;
          if (!response.ok || !data?.ok) {
            throw new Error(
              (data as { error?: string } | null)?.error ??
                "The mobile app could not reach the Flask server."
            );
          }

          setServerUrl(candidate);
          setServerMessage(
            `Connected to ${candidate}. Loaded model: ${data.modelFilename}. Classes: ${data.modelClasses.join(", ")}`
          );
          return;
        } catch (error) {
          lastError =
            error instanceof Error ? error.message : "Could not connect to the Flask server.";
        }
      }

      throw new Error(
        `${lastError} Tried: ${serverCandidates.join(", ")}. Make sure phone and laptop are on the same Wi-Fi.`
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not connect to the Flask server.";
      setServerMessage(message);
      Alert.alert("Server not reachable", message);
    } finally {
      setCheckingServer(false);
    }
  };

  const handleOpenBrowserUi = async () => {
    const targetServerUrl = serverCandidates[0] ?? "";
    if (!targetServerUrl) {
      Alert.alert("Server URL missing", "Enter your laptop server URL first.");
      return;
    }

    await Linking.openURL(targetServerUrl);
  };

  const handlePredict = async () => {
    const targetServerUrl = serverCandidates[0] ?? "";
    if (!targetServerUrl) {
      Alert.alert(
        "Server URL missing",
        "Enter your laptop URL, for example http://192.168.1.8:5008"
      );
      return;
    }

    if (!selectedImage) {
      Alert.alert("Image missing", "Choose a photo or take a picture first.");
      return;
    }

    setLoading(true);
    setPrediction(null);
    setServerMessage("Uploading image to the YOLOv8 server...");

    try {
      const formData = new FormData();
      formData.append("conf", confidence.toFixed(2));
      formData.append(
        "image",
        {
          uri: selectedImage.uri,
          name: selectedImage.fileName,
          type: selectedImage.mimeType,
        } as any
      );

      const response = await fetchJsonWithTimeout(`${targetServerUrl}/api/predict`, {
        method: "POST",
        body: formData,
      }, 20000);

      const data = (await parseResponseJson(response)) as PredictionResponse | null;
      if (!response.ok || !data?.ok) {
        throw new Error(
          (data as { error?: string } | null)?.error ?? "Prediction failed on the server."
        );
      }

      setPrediction(data);
      if (data.detectionRows.length > 0) {
        setServerMessage(
          `Prediction complete. Top stage: ${data.detectionRows[0].label} (${formatPercent(
            data.detectionRows[0].confidence
          )}).`
        );
      } else {
        setServerMessage(data.detectionHint ?? "Prediction complete with no visible detections.");
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Prediction failed on the server.";
      setServerMessage(message);
      Alert.alert("Prediction failed", message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        <ScrollView
          style={styles.flex}
          contentContainerStyle={styles.content}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.heroCard}>
            <Text style={styles.eyebrow}>YOLOv8 + Expo Go</Text>
            <Text style={styles.title}>Orchid stage detection on Android</Text>
            <Text style={styles.subtitle}>
              Pick a photo on your phone, send it to your laptop's Flask server, and see the
              annotated YOLO result right inside Expo Go.
            </Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>1. Server connection</Text>
            <Text style={styles.label}>Laptop server URL</Text>
            <TextInput
              value={serverUrl}
              onChangeText={setServerUrl}
              autoCapitalize="none"
              autoCorrect={false}
              placeholder="http://192.168.1.8:5008"
              keyboardType="url"
              style={styles.input}
            />
            <Text style={styles.helperText}>
              Start the Flask app on your laptop, then keep phone and laptop on the same Wi-Fi.
            </Text>
            {inferredServerCandidates[0] ? (
              <Text style={styles.helperText}>
                Auto-detected from Expo: {inferredServerCandidates[0]}
              </Text>
            ) : null}
            <View style={styles.inlineButtons}>
              <Pressable
                onPress={handleTestServer}
                disabled={checkingServer}
                style={({ pressed }) => [
                  styles.secondaryButton,
                  pressed && styles.buttonPressed,
                  checkingServer && styles.buttonDisabled,
                ]}
              >
                {checkingServer ? (
                  <ActivityIndicator color="#24524b" />
                ) : (
                  <Text style={styles.secondaryButtonText}>Test server</Text>
                )}
              </Pressable>
              <Pressable
                onPress={handleOpenBrowserUi}
                style={({ pressed }) => [
                  styles.secondaryButton,
                  pressed && styles.buttonPressed,
                ]}
              >
                <Text style={styles.secondaryButtonText}>Open browser UI</Text>
              </Pressable>
            </View>
            <Text style={styles.statusText}>{serverMessage}</Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>2. Choose image</Text>
            <View style={styles.inlineButtons}>
              <Pressable
                onPress={() => handleSelectImage("library")}
                style={({ pressed }) => [
                  styles.primaryButton,
                  pressed && styles.buttonPressed,
                ]}
              >
                <Text style={styles.primaryButtonText}>Pick from gallery</Text>
              </Pressable>
              <Pressable
                onPress={() => handleSelectImage("camera")}
                style={({ pressed }) => [
                  styles.primaryButton,
                  pressed && styles.buttonPressed,
                ]}
              >
                <Text style={styles.primaryButtonText}>Take photo</Text>
              </Pressable>
            </View>

            {selectedImage ? (
              <View style={styles.previewBlock}>
                <Image source={{ uri: previewImageUri as string }} style={styles.previewImage} />
                <Text style={styles.previewLabel}>{selectedImage.fileName}</Text>
                <Text style={styles.helperText}>
                  {selectedImage.width ?? "?"} x {selectedImage.height ?? "?"} |{" "}
                  {selectedImage.mimeType}
                </Text>
              </View>
            ) : (
              <Text style={styles.helperText}>No image selected yet.</Text>
            )}
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>3. Detection settings</Text>
            <Text style={styles.label}>Confidence threshold</Text>
            <TextInput
              value={confidenceText}
              onChangeText={setConfidenceText}
              keyboardType="decimal-pad"
              style={styles.input}
            />
            <Text style={styles.helperText}>
              Lower values show more boxes. Current value used: {confidence.toFixed(2)}
            </Text>
            <View style={styles.chipRow}>
              {QUICK_CONF_VALUES.map((value) => (
                <Pressable
                  key={value}
                  onPress={() => setConfidenceText(value)}
                  style={({ pressed }) => [
                    styles.chip,
                    value === confidence.toFixed(2) && styles.chipActive,
                    pressed && styles.buttonPressed,
                  ]}
                >
                  <Text
                    style={[
                      styles.chipText,
                      value === confidence.toFixed(2) && styles.chipTextActive,
                    ]}
                  >
                    {value}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>4. Run prediction</Text>
            <Pressable
              onPress={handlePredict}
              disabled={loading}
              style={({ pressed }) => [
                styles.predictButton,
                pressed && styles.buttonPressed,
                loading && styles.buttonDisabled,
              ]}
            >
              {loading ? (
                <ActivityIndicator color="#ffffff" />
              ) : (
                <Text style={styles.predictButtonText}>Send image to YOLOv8</Text>
              )}
            </Pressable>
            <Text style={styles.helperText}>
              Keep the Flask server running while sending predictions from the phone.
            </Text>
          </View>

          {prediction ? (
            <View style={styles.card}>
              <Text style={styles.cardTitle}>Prediction result</Text>
              <View style={styles.summaryBox}>
                <Text style={styles.summaryLabel}>Model</Text>
                <Text style={styles.summaryValue}>{prediction.modelFilename}</Text>
                <Text style={styles.summaryLabel}>Top result</Text>
                <Text style={styles.summaryValue}>
                  {topDetection
                    ? `${topDetection.label} (${formatPercent(topDetection.confidence)})`
                    : "No detections"}
                </Text>
                <Text style={styles.summaryLabel}>Image size</Text>
                <Text style={styles.summaryValue}>
                  {prediction.imageSize.width} x {prediction.imageSize.height}
                </Text>
              </View>

              {annotatedImageUri ? (
                <Image source={{ uri: annotatedImageUri }} style={styles.resultImage} />
              ) : null}

              {prediction.detectionHint ? (
                <View style={styles.infoBox}>
                  <Text style={styles.infoText}>{prediction.detectionHint}</Text>
                </View>
              ) : null}

              <Text style={styles.resultSectionTitle}>Detected stages</Text>
              {prediction.detectionRows.length > 0 ? (
                prediction.detectionRows.map((row) => (
                  <View key={row.label} style={styles.resultRow}>
                    <View style={styles.resultRowText}>
                      <Text style={styles.resultLabel}>{row.label}</Text>
                      <Text style={styles.helperText}>
                        {row.count} box(es) detected at this stage
                      </Text>
                    </View>
                    <View style={styles.confidencePill}>
                      <Text style={styles.confidencePillText}>{formatPercent(row.confidence)}</Text>
                    </View>
                  </View>
                ))
              ) : (
                <Text style={styles.helperText}>No classes passed the current threshold.</Text>
              )}

              {prediction.bestGuess ? (
                <View style={styles.bestGuessBox}>
                  <Text style={styles.bestGuessTitle}>Fallback estimate</Text>
                  <Text style={styles.bestGuessLabel}>{prediction.bestGuess.label}</Text>
                  <Text style={styles.helperText}>
                    Confidence {prediction.bestGuess.confidence.toFixed(4)} with support{" "}
                    {prediction.bestGuess.count}
                  </Text>
                </View>
              ) : null}

              {prediction.diagnostics ? (
                <View style={styles.diagnosticsBox}>
                  <Text style={styles.resultSectionTitle}>Diagnostics</Text>
                  <Text style={styles.helperText}>
                    Selected-threshold boxes: {prediction.diagnostics.boxes_at_selected_conf ?? 0}
                  </Text>
                  {prediction.diagnostics.boxes_at_probe_conf != null ? (
                    <Text style={styles.helperText}>
                      Probe boxes: {prediction.diagnostics.boxes_at_probe_conf}
                    </Text>
                  ) : null}
                  {prediction.diagnostics.max_conf_at_probe != null ? (
                    <Text style={styles.helperText}>
                      Probe max confidence: {prediction.diagnostics.max_conf_at_probe}
                    </Text>
                  ) : null}
                  {prediction.diagnostics.adaptive_conf_used != null ? (
                    <Text style={styles.helperText}>
                      Adaptive confidence used: {prediction.diagnostics.adaptive_conf_used}
                    </Text>
                  ) : null}
                </View>
              ) : null}
            </View>
          ) : null}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#f3f6fb",
  },
  flex: {
    flex: 1,
  },
  content: {
    padding: 16,
    paddingBottom: 32,
  },
  heroCard: {
    backgroundColor: "#11253d",
    borderRadius: 20,
    padding: 20,
    marginBottom: 16,
  },
  eyebrow: {
    color: "#8fd3ff",
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 8,
  },
  title: {
    color: "#ffffff",
    fontSize: 28,
    fontWeight: "800",
    marginBottom: 10,
  },
  subtitle: {
    color: "#d5e7ff",
    fontSize: 15,
    lineHeight: 22,
  },
  card: {
    backgroundColor: "#ffffff",
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#dce5ef",
    shadowColor: "#10233a",
    shadowOpacity: 0.07,
    shadowOffset: { width: 0, height: 6 },
    shadowRadius: 12,
    elevation: 2,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#14253d",
    marginBottom: 12,
  },
  label: {
    fontSize: 14,
    fontWeight: "600",
    color: "#223a56",
    marginBottom: 6,
  },
  input: {
    borderWidth: 1,
    borderColor: "#c9d7e6",
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
    backgroundColor: "#fbfdff",
    fontSize: 15,
    color: "#13263f",
  },
  helperText: {
    marginTop: 8,
    color: "#5b6b7d",
    fontSize: 13,
    lineHeight: 18,
  },
  statusText: {
    marginTop: 12,
    fontSize: 13,
    lineHeight: 19,
    color: "#24524b",
    backgroundColor: "#edf8f4",
    borderRadius: 12,
    padding: 12,
  },
  inlineButtons: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginTop: 12,
  },
  primaryButton: {
    backgroundColor: "#1d6ef2",
    borderRadius: 12,
    paddingVertical: 13,
    paddingHorizontal: 16,
    marginRight: 10,
    marginBottom: 10,
  },
  primaryButtonText: {
    color: "#ffffff",
    fontSize: 14,
    fontWeight: "700",
  },
  secondaryButton: {
    backgroundColor: "#eef5fb",
    borderRadius: 12,
    paddingVertical: 13,
    paddingHorizontal: 16,
    marginRight: 10,
    marginBottom: 10,
    minWidth: 132,
    alignItems: "center",
    justifyContent: "center",
  },
  secondaryButtonText: {
    color: "#24524b",
    fontSize: 14,
    fontWeight: "700",
  },
  predictButton: {
    backgroundColor: "#0f8a5f",
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 16,
  },
  predictButtonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "800",
  },
  buttonPressed: {
    opacity: 0.86,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginTop: 10,
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 999,
    backgroundColor: "#edf3fb",
    marginRight: 10,
    marginBottom: 10,
  },
  chipActive: {
    backgroundColor: "#d9ecff",
    borderWidth: 1,
    borderColor: "#8ebeff",
  },
  chipText: {
    color: "#335270",
    fontWeight: "700",
  },
  chipTextActive: {
    color: "#0f57be",
  },
  previewBlock: {
    marginTop: 12,
  },
  previewImage: {
    width: "100%",
    height: 220,
    borderRadius: 16,
    backgroundColor: "#ebf1f7",
  },
  previewLabel: {
    marginTop: 10,
    fontSize: 14,
    fontWeight: "700",
    color: "#17304c",
  },
  summaryBox: {
    backgroundColor: "#f8fbff",
    borderRadius: 14,
    padding: 14,
    marginBottom: 14,
  },
  summaryLabel: {
    color: "#5f7083",
    fontSize: 12,
    textTransform: "uppercase",
    fontWeight: "700",
    marginBottom: 4,
  },
  summaryValue: {
    color: "#13263f",
    fontSize: 16,
    fontWeight: "700",
    marginBottom: 10,
  },
  resultImage: {
    width: "100%",
    height: 280,
    borderRadius: 16,
    backgroundColor: "#ebf1f7",
    marginBottom: 14,
  },
  infoBox: {
    backgroundColor: "#fff9e9",
    borderRadius: 14,
    padding: 12,
    marginBottom: 14,
  },
  infoText: {
    color: "#755b08",
    fontSize: 13,
    lineHeight: 19,
  },
  resultSectionTitle: {
    color: "#17304c",
    fontSize: 16,
    fontWeight: "800",
    marginBottom: 10,
  },
  resultRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#eef3f8",
  },
  resultRowText: {
    flex: 1,
    paddingRight: 10,
  },
  resultLabel: {
    color: "#13263f",
    fontSize: 15,
    fontWeight: "700",
  },
  confidencePill: {
    backgroundColor: "#e8f5ee",
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
  },
  confidencePillText: {
    color: "#106a45",
    fontWeight: "800",
  },
  bestGuessBox: {
    marginTop: 16,
    backgroundColor: "#f5efff",
    borderRadius: 14,
    padding: 14,
  },
  bestGuessTitle: {
    color: "#65489b",
    fontSize: 13,
    fontWeight: "800",
    textTransform: "uppercase",
    marginBottom: 8,
  },
  bestGuessLabel: {
    color: "#4f3780",
    fontSize: 19,
    fontWeight: "800",
  },
  diagnosticsBox: {
    marginTop: 16,
    backgroundColor: "#f8fbff",
    borderRadius: 14,
    padding: 14,
  },
});
