"use strict";
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { autoUpdater } = require("electron-updater");
const { exec } = require('child_process');
const path = require('path');

if (process.env.NODE_ENV === 'development') {
    // Enable live reload only in development
    try {
        require('electron-reload')(path.join(__dirname, '../../'), {
            electron: path.join(__dirname, '../../node_modules/.bin/electron'),
        });
        console.log("Electron reload enabled for development.");
    } catch (err) {
        console.error("Failed to load electron-reload:", err);
    }
}

console.log("Current working directory:", process.cwd());

let mainWindow;
let editorWindow;

function createWindow() {
    // Create main window (index.html)
    mainWindow = new BrowserWindow({
        width: 1050,
        height: 800,
        minWidth: 800,
        minHeight: 580,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        },
    });

    // Load the index.html page for the main window
    mainWindow.loadFile(path.join(__dirname, '../pages/0-home/index.html'));  // Corrected path to index.html

    // Check for updates
    autoUpdater.checkForUpdatesAndNotify();

    autoUpdater.on("update-available", () => {
        mainWindow.webContents.send("update_available");
    });

    autoUpdater.on("update-downloaded", () => {
        mainWindow.webContents.send("update_downloaded");
    });
}

function createEditorWindow() {
    if (editorWindow) return;  // Prevent multiple editor windows

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

    // Load the editor page (e.g., chat.html)
    editorWindow.loadFile('chat.html');

    editorWindow.on('closed', () => {
        editorWindow = null;
    });
}

function startAiServer() {
    const pythonScript = path.join(__dirname, '..', 'ai_server.py');  // Path to ai_server.py
    const command = `python "${pythonScript}"`;  // Run the Python script using the global Python

    exec(command, (error, stdout, stderr) => {
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
    createWindow();    // Load the main window
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});

// Handle file open and save dialogs
ipcMain.handle('dialog:open', async () => {
    const result = await dialog.showOpenDialog({
        title: 'Open a note file',
        filters: [{ name: 'Text Files', extensions: ['txt', 'md'] }],
    });

    if (result.filePaths && result.filePaths.length > 0) {
        return result.filePaths[0];
    }
    return null;
});

ipcMain.handle('dialog:save', async (_event, currentFilePath) => {
    const defaultPath = currentFilePath || path.join(app.getPath('documents'), 'newfile.txt');
    const result = await dialog.showSaveDialog({
        title: 'Save your text file',
        defaultPath,
        filters: [{ name: 'Text Files', extensions: ['txt', 'md'] }],
    });

    if (result.filePath) {
        return result.filePath;
    }
    return null;
});

ipcMain.handle('dialog:openDirectory', async () => {
    const result = await dialog.showOpenDialog({
        properties: ['openDirectory'],
    });

    if (result.filePaths && result.filePaths.length > 0) {
        return result.filePaths[0];
    }
    return null;
});

// Handle update events from renderer
ipcMain.on("restart_app", () => {
    autoUpdater.quitAndInstall();
});
