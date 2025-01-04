import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import { exec } from 'child_process';
import path from 'path';
import fs from 'fs';

// Enable live reload
require('electron-reload')(path.join(__dirname, '../'), {
  electron: path.join(__dirname, '../node_modules/.bin/electron'),
});

// Log the current working directory
console.log("Current working directory:", process.cwd());

let mainWindow: BrowserWindow | null;
let editorWindow: BrowserWindow | null;

function createWindow() {
  // Create main window (index.html)
  mainWindow = new BrowserWindow({
    width: 1050,
    height: 800,
    minWidth: 500,
    minHeight: 580,
    webPreferences: {
      nodeIntegration: true,  // Ensure Node integration is enabled for access to Node.js
      contextIsolation: false, // Disable context isolation for direct use of Node.js modules
    },
  });

  // Load the index.html page for the main window
  mainWindow.loadFile('index.html');
}

// Function to create a separate editor window (test.html or chat.html)
function createEditorWindow() {
  if (editorWindow) return; // Prevent multiple editor windows

  editorWindow = new BrowserWindow({
    width: 1050,
    height: 800,
    minWidth: 500,
    minHeight: 580,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  // Load the test.html page for text editing
  editorWindow.loadFile('chat.html');

  // Handle the editor window close
  editorWindow.on('closed', () => {
    editorWindow = null; // Reset editor window reference
  });
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
  createWindow();    // Load the main window (index.html)

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Handle file open dialog request
ipcMain.handle('dialog:open', async () => {
  // Open file dialog
  const result = await dialog.showOpenDialog({
    title: 'Open a note file',
    filters: [{ name: 'Text Files', extensions: ['txt', 'md'] }],
  });

  // Check if result is an object with filePaths or a simple string array
  if ('filePaths' in result && Array.isArray(result.filePaths)) {
    if (result.filePaths.length > 0) {
      return result.filePaths[0];  // Return the first file path
    }
  } else if (Array.isArray(result)) {
    // In case the result is directly a string array (older Electron versions)
    if (result.length > 0) {
      return result[0];  // Return the first file path
    }
  }

  return null;  // Return null if no file is selected
});

// Handle file save dialog request
ipcMain.handle('dialog:save', async (_event, currentFilePath: string | null) => {
  // Set the default path to the current file path if available
  const defaultPath = currentFilePath || path.join(app.getPath('documents'), 'newfile.txt');

  const result: any = await dialog.showSaveDialog({
    title: 'Save your text file',
    defaultPath,
    filters: [{ name: 'Text Files', extensions: ['txt', 'md'] }],
  });

  if (!result.canceled && result.filePath) {
    return result.filePath;
  }
  return null;
});

// Handle folder open dialog request for reminders.html
ipcMain.handle('dialog:openDirectory', async () => {
  // Open folder dialog
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'] // Allow selecting a directory
  });

  if ('filePaths' in result && Array.isArray(result.filePaths)) {
    if (result.filePaths.length > 0) {
      return result.filePaths[0];  // Return the first selected folder path
    }
  }

  return null; // Return null if no folder is selected
});

// Quit the app when all windows are closed (for macOS)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
