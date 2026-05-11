# Orchid YOLO Mobile (Expo Go)

This Expo app sends an image from your Android phone to the YOLOv8 Flask server in `../web_inference` and shows:

- the annotated prediction image
- the detected orchid stages
- confidence scores
- fallback hints when confidence is low

## Folder

This app lives in:

`YoloVersion_Check/YOLOv8/orchid-yolo-mobile`

## 1. Start the Flask server on your laptop

Open a terminal in:

`YoloVersion_Check/YOLOv8/web_inference`

Run:

```powershell
python app.py
```

The server now listens on port `5008` and is reachable on your local network.

If Windows Firewall asks for permission, allow it on your private network.

## 2. Install mobile app dependencies

Open a second terminal in:

`YoloVersion_Check/YOLOv8/orchid-yolo-mobile`

Run:

```powershell
npm install
```

## 3. Start Expo

```powershell
npm run start
```

Important:

- use `LAN` mode in Expo
- keep the phone and laptop on the same Wi-Fi
- open the project in the Expo Go app by scanning the QR code

## 4. Use the app on Android

1. In Expo Go, open the app.
2. The server URL should auto-fill from the Metro host. If it does not, enter your laptop IP manually, for example:

   `http://192.168.1.8:5008`

3. Tap `Test server`.
4. Tap `Pick from gallery` or `Take photo`.
5. Tap `Send image to YOLOv8`.

## Notes

- Expo Go cannot run the Python YOLOv8 model directly on the phone. The phone app uploads the image to your Flask server for inference.
- The Flask app now exposes:
  - `GET /api/health`
  - `POST /api/predict`
- The original browser UI still works.

## Troubleshooting

### Phone cannot connect to server

- Make sure the Flask terminal is still running.
- Make sure both devices are on the same Wi-Fi.
- Check your laptop IP with:

```powershell
ipconfig
```

- Then use `http://YOUR_LAPTOP_IP:5008` inside the app.

### Expo command not found

Install Node.js LTS first, then run:

```powershell
npm install
```

## Main files

- `App.tsx` - full mobile UI and upload flow
- `app.json` - Expo config
- `package.json` - dependencies and scripts
- `../web_inference/app.py` - YOLOv8 browser + mobile API server
