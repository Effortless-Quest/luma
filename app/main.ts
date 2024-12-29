import { app, BrowserWindow } from 'electron';
import { exec } from 'child_process';
import path from 'path';

let mainWindow: BrowserWindow | null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,  // Ensure Node integration is enabled for access to Node.js
    },
  });

  mainWindow.loadFile('index.html');
}

// Run the Python AI server when the Electron app starts
function startAiServer() {
  const pythonScript = path.join(__dirname, '..', 'ai_server.py');  // Path to ai_server.py outside of the app folder

  // Use the global Python installation (assuming 'python' is in your system's PATH)
  const command = `python "${pythonScript}"`;  // Run the Python script using the global Python

  // Run Python server in the background
  exec(command, (error: Error | null, stdout: string, stderr: string) => {
    if (error) {
      console.error(`Error starting AI server: ${error.message}`);
      return;
    }
    if (stderr) {
      console.error(`stderr: ${stderr}`);
      return;
    }
    console.log(`stdout: ${stdout}`);
  });
}

app.whenReady().then(() => {
  startAiServer();  // Start the Python server when Electron app is ready
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit the app when all windows are closed (for macOS)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
