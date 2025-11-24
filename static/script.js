// Medical Chatbot Pro - Main JavaScript

const chatContainer = document.getElementById('chatContainer');
const messagesDiv = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const typingIndicator = document.getElementById('typingIndicator');
const welcomeMessage = document.getElementById('welcomeMessage');

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});

// Send on Enter (Shift+Enter for new line)
messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function formatText(text) {
    // Format bold text
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Format numbered lists
    text = text.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
    
    // Wrap lists in ol tags
    text = text.replace(/(<li>.*?<\/li>\s*)+/gs, '<ol>$&</ol>');
    
    // Format paragraphs
    text = text.split('\n\n').map(p => p.trim() ? `<p>${p}</p>` : '').join('');
    
    return text;
}

function addMessage(text, isUser, sources = null) {
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = isUser ? '👤' : '🤖';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    // Format bot messages with HTML
    if (!isUser) {
        content.innerHTML = formatText(text);
        
        // Add copy button
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-button';
        copyBtn.textContent = 'Copy';
        copyBtn.onclick = () => copyToClipboard(text, copyBtn);
        content.appendChild(copyBtn);
    } else {
        content.textContent = text;
    }

    // Add sources
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        
        const sourcesTitle = document.createElement('div');
        sourcesTitle.className = 'sources-title';
        sourcesTitle.innerHTML = '📚 Sources Used:';
        sourcesDiv.appendChild(sourcesTitle);

        sources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            const sourceNum = document.createElement('span');
            sourceNum.className = 'source-number';
            sourceNum.textContent = index + 1;
            
            const sourceText = document.createElement('span');
            sourceText.textContent = source.source.split('\\').pop();
            
            sourceItem.appendChild(sourceNum);
            sourceItem.appendChild(sourceText);
            sourcesDiv.appendChild(sourceItem);
        });

        content.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    messagesDiv.appendChild(messageDiv);
    
    scrollToBottom();
}

function addErrorMessage(text) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = '⚠️ ' + text;
    messagesDiv.appendChild(errorDiv);
    scrollToBottom();
}

function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        const originalText = button.textContent;
        button.textContent = '✓ Copied';
        button.style.background = 'rgba(72, 187, 120, 0.2)';
        button.style.color = '#48bb78';
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.background = '';
            button.style.color = '';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        button.textContent = '✗ Failed';
    });
}

function clearChat() {
    if (confirm('Clear all messages? This cannot be undone.')) {
        messagesDiv.innerHTML = '';
        welcomeMessage.style.display = 'block';
    }
}

async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    addMessage(message, true);
    
    messageInput.value = '';
    messageInput.style.height = 'auto';
    messageInput.disabled = true;
    sendButton.disabled = true;
    
    typingIndicator.classList.add('active');
    scrollToBottom();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        typingIndicator.classList.remove('active');
        
        if (response.ok) {
            addMessage(data.answer, false, data.sources);
        } else {
            addErrorMessage(data.error || 'Failed to get response. Please try again.');
        }
    } catch (error) {
        typingIndicator.classList.remove('active');
        addErrorMessage('Network error. Please check your connection and try again.');
        console.error('Error:', error);
    } finally {
        messageInput.disabled = false;
        sendButton.disabled = false;
        messageInput.focus();
    }
}

function askQuestion(question) {
    const cleanQuestion = question.replace(/[^\w\s?.,]/gi, '').trim();
    messageInput.value = cleanQuestion;
    sendMessage();
}

// Focus on input when page loads
window.addEventListener('load', function() {
    messageInput.focus();
});