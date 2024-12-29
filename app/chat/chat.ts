// Chat logic
const chatBox = document.getElementById('chat-box')! as HTMLDivElement;
const userInput = document.getElementById('user-input')! as HTMLInputElement;
const sendBtn = document.getElementById('send-btn')! as HTMLButtonElement;

// Function to send message to the backend
async function sendMessage(message: string) {
    const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
    });

    const data = await response.json();
    return data.response;
}

// Event listener for the send button
sendBtn.addEventListener('click', async () => {
    const userMessage = userInput.value.trim();
    if (!userMessage) return;

    // Display user message in chat box
    const userMessageElem = document.createElement('div');
    userMessageElem.textContent = `You: ${userMessage}`;
    chatBox.appendChild(userMessageElem);

    // Send to AI server and display AI response
    try {
        const aiResponse = await sendMessage(userMessage);
        const aiMessageElem = document.createElement('div');
        aiMessageElem.textContent = `AI: ${aiResponse}`;
        chatBox.appendChild(aiMessageElem);
    } catch (error) {
        const errorMessageElem = document.createElement('div');
        errorMessageElem.textContent = 'Error: Could not connect to the server';
        chatBox.appendChild(errorMessageElem);
    }

    // Clear user input
    userInput.value = '';

    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;
});
