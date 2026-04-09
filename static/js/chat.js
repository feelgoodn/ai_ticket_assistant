(function () {
    const params = new URLSearchParams(window.location.search);
    const token  = params.get('token');

    if (token) {
        localStorage.setItem('idmworks_token', token);
        window.history.replaceState({}, document.title, window.location.pathname);
    }

    if (!localStorage.getItem('idmworks_token')) {
        window.location.href = 'http://localhost:5001/';
    }
})();


class SupportChatApp {
    constructor() {
        this.messagesContainer = document.getElementById('messages');
        this.messagesWrapper   = document.getElementById('messagesWrapper');
        this.userInput         = document.getElementById('userInput');
        this.sendButton        = document.getElementById('sendButton');
        this.charCount         = document.getElementById('charCount');
        this.scrollButton      = document.getElementById('scrollToBottom');
        this.sessionId         = this._generateSessionId();
        this.isLoading         = false;

        this._init();
    }

    _init() {
        this.sendButton.addEventListener('click',  () => this._handleSend());
        this.userInput.addEventListener('keydown', (e) => this._handleKeyPress(e));
        this.userInput.addEventListener('input',   () => {
            this._updateCharCount();
            this._autoResize();
        });
        this.messagesWrapper.addEventListener('scroll', () => this._handleScroll());
        if (this.scrollButton) {
            this.scrollButton.addEventListener('click', () => this._scrollToBottom(true));
        }
        setTimeout(() => this.userInput.focus(), 300);
    }

    _generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).slice(2, 11);
    }

    _handleKeyPress(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this._handleSend();
        }
    }

    _updateCharCount() {
        const len = this.userInput.value.length;
        if (this.charCount) {
            this.charCount.textContent = len;
            this.charCount.style.color =
                len > 1800 ? '#f87171' :
                len > 1500 ? '#fbbf24' : '';
        }
    }

    _autoResize() {
        this.userInput.style.height = 'auto';
        this.userInput.style.height = Math.min(this.userInput.scrollHeight, 150) + 'px';
    }

    async _handleSend() {
        const message = this.userInput.value.trim();
        if (!message || this.isLoading) return;

        const welcomeCard = document.querySelector('.welcome-card');
        if (welcomeCard) {
            welcomeCard.style.opacity   = '0';
            welcomeCard.style.transform = 'scale(0.95)';
            setTimeout(() => welcomeCard.remove(), 300);
        }

        this._addMessage(message, 'user');
        this.userInput.value = '';
        this._updateCharCount();
        this._autoResize();
        this._setLoading(true);
        this._showTyping();

        try {
            const data = await this._post('/api/chat', {
                message:    message,
                session_id: this.sessionId,
            });
            this._removeTyping();

            // FIXED: check data.response not data.success
            if (data && data.response) {
                this._addMessage(data.response, 'assistant');
            } else {
                this._addMessage('Sorry, something went wrong. Please try again.', 'assistant');
            }
        } catch (err) {
            console.error('Chat error:', err);
            this._removeTyping();
            this._addMessage("Couldn't reach the server. Please try again.", 'assistant');
        } finally {
            this._setLoading(false);
        }
    }

    // UPDATED: sends JWT token with every request
    async _post(url, body) {
        const token = localStorage.getItem('idmworks_token');

        if (!token) {
            window.location.href = 'http://localhost:5001/';
            return;
        }

        const resp = await fetch(url, {
            method:  'POST',
            headers: {
                'Content-Type':  'application/json',
                'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify(body),
        });

        // Token expired → back to login
        if (resp.status === 401) {
            localStorage.removeItem('idmworks_token');
            window.location.href = 'http://localhost:5001/';
            return;
        }

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json();
    }

    _addMessage(content, role) {
        const group = document.createElement('div');
        group.className       = `message-group ${role}`;
        group.style.opacity   = '0';
        group.style.transform = 'translateY(16px)';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = role === 'assistant' ? this._botSvg() : this._userSvg();

        const bubble = document.createElement('div');
        bubble.className = 'message';

        const content_div = document.createElement('div');
        content_div.className = 'message-content';
        content_div.innerHTML = this._format(content);

        bubble.appendChild(content_div);
        group.appendChild(avatar);
        group.appendChild(bubble);
        this.messagesContainer.appendChild(group);

        requestAnimationFrame(() => {
            group.style.transition = 'all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1)';
            group.style.opacity    = '1';
            group.style.transform  = 'translateY(0)';
        });

        this._scrollToBottom();
    }

    _format(text) {
        const escaped = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        let formatted = escaped
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g,     '<em>$1</em>');

        const lines  = formatted.split('\n');
        const result = [];
        let inList   = false;

        for (const raw of lines) {
            const line = raw.trim();
            if (!line) {
                if (inList) { result.push('</ul>'); inList = false; }
                continue;
            }
            if (line.startsWith('•') || line.startsWith('-')) {
                if (!inList) { result.push('<ul>'); inList = true; }
                result.push(`<li>${line.slice(1).trim()}</li>`);
            } else {
                if (inList) { result.push('</ul>'); inList = false; }
                result.push(`<p>${line}</p>`);
            }
        }
        if (inList) result.push('</ul>');
        return result.join('');
    }

    _showTyping() {
        const indicator = document.createElement('div');
        indicator.className     = 'message-group assistant';
        indicator.id            = 'typing-indicator';
        indicator.style.opacity = '0';
        indicator.innerHTML = `
            <div class="message-avatar">${this._botSvg()}</div>
            <div class="message">
                <div class="message-content">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>`;
        this.messagesContainer.appendChild(indicator);
        requestAnimationFrame(() => {
            indicator.style.transition = 'opacity 0.25s';
            indicator.style.opacity    = '1';
        });
        this._scrollToBottom();
    }

    _removeTyping() {
        const el = document.getElementById('typing-indicator');
        if (el) {
            el.style.opacity = '0';
            setTimeout(() => el.remove(), 250);
        }
    }

    _setLoading(loading) {
        this.isLoading           = loading;
        this.sendButton.disabled = loading;
        this.userInput.disabled  = loading;
        this.sendButton.classList.toggle('loading', loading);
    }

    _scrollToBottom(smooth = true) {
        setTimeout(() => {
            this.messagesWrapper.scrollTo({
                top:      this.messagesWrapper.scrollHeight,
                behavior: smooth ? 'smooth' : 'auto',
            });
        }, 80);
    }

    _handleScroll() {
        if (!this.scrollButton) return;
        const { scrollTop, scrollHeight, clientHeight } = this.messagesWrapper;
        this.scrollButton.style.display =
            scrollHeight - scrollTop - clientHeight > 120 ? 'flex' : 'none';
    }

    _botSvg() {
        return `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2.5"/>
            <path d="M8 14C8 14 9.5 16 12 16C14.5 16 16 14 16 14"
                  stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="9"  cy="9" r="1.5" fill="currentColor"/>
            <circle cx="15" cy="9" r="1.5" fill="currentColor"/>
        </svg>`;
    }

    _userSvg() {
        return `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="8" r="4" stroke="currentColor" stroke-width="2.5"/>
            <path d="M4 20C4 16.6863 6.68629 14 10 14H14C17.3137 14 20 16.6863 20 20"
                  stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
        </svg>`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new SupportChatApp();
});