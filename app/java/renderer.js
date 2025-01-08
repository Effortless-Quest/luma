const { ipcRenderer } = require("electron");

ipcRenderer.on("update_available", () => {
    alert("A new update is available. Downloading now...");
});

ipcRenderer.on("update_downloaded", () => {
    const shouldRestart = confirm("Update downloaded. Restart now?");
    if (shouldRestart) {
        ipcRenderer.send("restart_app");
    }
});
