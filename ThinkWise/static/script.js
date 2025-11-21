// static/script.js
// Advanced Concept Breakdown System with AI-Powered Learning Features

class ThinkWiseApp {
    constructor() {
        this.state = {
            currentSession: null,
            breakdownHistory: [],
            userPreferences: {},
            isAssistantOpen: false,
            voiceRecognition: null,
            sessionStartTime: null,
            activeRequests: new Map(),
            cache: new Map(),
            retryCount: 0,
            maxRetries: 3
        };
        
        this.init();
    }

    init() {
        this.bindElements();
        this.setupEventListeners();
        this.loadUserPreferences();
        this.setupVoiceRecognition();
        this.startSessionTimer();
        this.setupRealTimeSuggestions();
        this.setupKeyboardShortcuts();
        this.loadBreakdownHistory();
    }

    bindElements() {
        this.elements = {
            form: document.getElementById("concept-form"),
            topicInput: document.getElementById("topic-input"),
            complexitySelect: document.getElementById("complexity-select"),
            learningStyleSelect: document.getElementById("learning-style-select"),
            submitButton: document.querySelector(".submit-button"),
            spinner: document.querySelector(".spinner"),
            resultsSection: document.getElementById("results-section"),
            resultsContainer: document.getElementById("concept-results"),
            suggestionsPanel: document.getElementById("suggestions-panel"),
            voiceInputBtn: document.getElementById("voice-input-btn"),
            assistantToggle: document.getElementById("assistant-toggle"),
            assistantWindow: document.getElementById("assistant-window"),
            assistantMessages: document.getElementById("assistant-messages"),
            assistantInput: document.getElementById("assistant-input"),
            assistantSend: document.getElementById("assistant-send"),
            sessionTimer: document.getElementById("timer-display"),
            exportBtn: document.getElementById("export-breakdown"),
            quizBtn: document.getElementById("generate-quiz"),
            notesBtn: document.getElementById("save-notes"),
            shareBtn: document.getElementById("share-breakdown")
        };

        // Validate critical elements
        const criticalElements = ['form', 'topicInput', 'complexitySelect', 'submitButton', 'spinner', 'resultsSection', 'resultsContainer'];
        const missingElements = criticalElements.filter(el => !this.elements[el]);
        
        if (missingElements.length > 0) {
            console.warn("Missing required DOM elements:", missingElements);
            this.showNotification("Application initialization failed. Please refresh the page.", "error");
            return;
        }
    }

    setupEventListeners() {
        // Form submission
        this.elements.form.addEventListener("submit", (e) => this.handleConceptBreakdown(e));

        // Quick topic badges
        document.querySelectorAll('.interactive-badge').forEach(badge => {
            badge.addEventListener('click', () => {
                const topic = badge.getAttribute('data-topic');
                this.elements.topicInput.value = topic;
                this.elements.form.dispatchEvent(new Event('submit'));
            });
        });

        // Voice input
        if (this.elements.voiceInputBtn) {
            this.elements.voiceInputBtn.addEventListener('click', () => this.toggleVoiceInput());
        }

        // Assistant functionality
        if (this.elements.assistantToggle) {
            this.elements.assistantToggle.addEventListener('click', () => this.toggleAssistant());
        }
        if (this.elements.assistantSend) {
            this.elements.assistantSend.addEventListener('click', () => this.sendAssistantMessage());
        }
        if (this.elements.assistantInput) {
            this.elements.assistantInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.sendAssistantMessage();
            });
        }

        // Results actions
        if (this.elements.exportBtn) {
            this.elements.exportBtn.addEventListener('click', () => this.exportBreakdown());
        }
        if (this.elements.quizBtn) {
            this.elements.quizBtn.addEventListener('click', () => this.generateQuiz());
        }
        if (this.elements.notesBtn) {
            this.elements.notesBtn.addEventListener('click', () => this.saveToNotes());
        }
        if (this.elements.shareBtn) {
            this.elements.shareBtn.addEventListener('click', () => this.shareBreakdown());
        }

        // Input validation
        this.elements.topicInput.addEventListener('input', () => this.validateTopicInput());
    }

    setupRealTimeSuggestions() {
        let timeoutId;
        
        this.elements.topicInput.addEventListener('input', () => {
            clearTimeout(timeoutId);
            const query = this.elements.topicInput.value.trim();
            
            if (query.length < 2) {
                this.hideSuggestions();
                return;
            }
            
            timeoutId = setTimeout(() => {
                this.fetchSuggestions(query);
            }, 300);
        });
        
        // Hide suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.elements.suggestionsPanel?.contains(e.target) && e.target !== this.elements.topicInput) {
                this.hideSuggestions();
            }
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to submit
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                if (this.elements.form && document.activeElement === this.elements.topicInput) {
                    this.elements.form.dispatchEvent(new Event('submit'));
                }
            }
            
            // Escape to clear input or close panels
            if (e.key === 'Escape') {
                if (this.elements.suggestionsPanel && !this.elements.suggestionsPanel.classList.contains('hidden')) {
                    this.hideSuggestions();
                } else if (this.elements.topicInput.value) {
                    this.elements.topicInput.value = '';
                }
            }
            
            // Ctrl/Cmd + K to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.elements.topicInput.focus();
            }
        });
    }

    setupVoiceRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            if (this.elements.voiceInputBtn) {
                this.elements.voiceInputBtn.style.display = 'none';
            }
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.state.voiceRecognition = new SpeechRecognition();
        
        this.state.voiceRecognition.continuous = false;
        this.state.voiceRecognition.interimResults = false;
        this.state.voiceRecognition.lang = 'en-US';

        this.state.voiceRecognition.onstart = () => {
            this.state.isListening = true;
            this.elements.voiceInputBtn.classList.add('listening');
            this.showNotification("Listening... Speak now", "info");
        };

        this.state.voiceRecognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.elements.topicInput.value = transcript;
            this.showNotification(`Heard: "${transcript}"`, "success");
        };

        this.state.voiceRecognition.onerror = (event) => {
            console.error('Speech recognition error', event.error);
            this.showNotification(`Voice input failed: ${event.error}`, "error");
        };

        this.state.voiceRecognition.onend = () => {
            this.state.isListening = false;
            this.elements.voiceInputBtn.classList.remove('listening');
        };
    }

    async handleConceptBreakdown(event) {
        event.preventDefault();
        
        const topic = this.elements.topicInput.value.trim();
        const complexity = this.elements.complexitySelect.value;
        const learningStyle = this.elements.learningStyleSelect?.value || "visual";

        if (!this.validateTopicInput()) {
            return;
        }

        // Check cache first
        const cacheKey = this.generateCacheKey(topic, complexity, learningStyle);
        const cached = this.getCachedResult(cacheKey);
        
        if (cached) {
            this.showNotification("Loading from cache...", "info");
            this.renderConceptBreakdown(cached.data, cached.metadata);
            return;
        }

        this.setLoadingState(true);
        this.state.retryCount = 0;

        try {
            await this.fetchBreakdownWithRetry(topic, complexity, learningStyle);
        } catch (error) {
            console.error("Final attempt failed:", error);
            this.showNotification("Failed to generate breakdown after multiple attempts", "error");
            this.setLoadingState(false);
        }
    }

    async fetchBreakdownWithRetry(topic, complexity, learningStyle) {
        const controller = new AbortController();
        const requestId = Date.now().toString();
        
        this.state.activeRequests.set(requestId, controller);

        while (this.state.retryCount <= this.state.maxRetries) {
            try {
                const response = await this.fetchBreakdown(topic, complexity, learningStyle, controller.signal);
                
                if (response.success && response.data) {
                    const metadata = {
                        topic,
                        complexity,
                        learningStyle,
                        ...response.metadata
                    };
                    
                    // Cache successful response
                    this.cacheResult(this.generateCacheKey(topic, complexity, learningStyle), {
                        data: response.data,
                        metadata
                    });
                    
                    this.renderConceptBreakdown(response.data, metadata);
                    this.trackUserActivity('concept_breakdown', topic, complexity);
                    this.state.activeRequests.delete(requestId);
                    return;
                } else {
                    throw new Error("Invalid response format");
                }

            } catch (error) {
                this.state.retryCount++;
                
                if (this.state.retryCount <= this.state.maxRetries) {
                    const delay = Math.min(1000 * Math.pow(2, this.state.retryCount), 10000);
                    console.log(`Retry ${this.state.retryCount} after ${delay}ms`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                } else {
                    this.state.activeRequests.delete(requestId);
                    throw error;
                }
            }
        }
    }

    async fetchBreakdown(topic, complexity, learningStyle, signal) {
        const response = await fetch("/api/v1/breakdown", {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "X-Request-ID": this.generateRequestId()
            },
            body: JSON.stringify({ 
                topic, 
                complexity, 
                learning_style: learningStyle,
                session_id: this.getSessionId()
            }),
            signal
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.error || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    renderConceptBreakdown(data, metadata) {
        if (!data) {
            this.showNotification('No data returned from AI service', 'error');
            return;
        }

        const html = this.generateEnhancedBreakdownHTML(data, metadata);
        this.elements.resultsContainer.innerHTML = html;
        this.elements.resultsSection.classList.remove("hidden");

        // Add interactive functionality
        this.setupInteractiveElements();
        
        // Animate results appearance
        this.animateResultsAppearance();
        
        // Save to history
        this.saveToHistory(data, metadata);
        
        // Update related suggestions
        if (data.suggested_next_topics) {
            this.updateRelatedSuggestions(data.suggested_next_topics);
        }

        // Scroll to results
        this.elements.resultsSection.scrollIntoView({ 
            behavior: "smooth", 
            block: "start"
        });
    }

    generateEnhancedBreakdownHTML(data, metadata) {
        return `
            <div class="concept-breakdown enhanced-breakdown" data-topic="${this.escapeHtml(metadata.topic)}">
                <div class="breakdown-header">
                    <h2>${this.escapeHtml(metadata.topic)}</h2>
                    <div class="breakdown-meta">
                        <span class="complexity-badge">${metadata.complexity}</span>
                        <span class="learning-style-badge">${metadata.learningStyle}</span>
                        <span class="generated-time">${new Date().toLocaleTimeString()}</span>
                    </div>
                </div>

                ${this.renderExpandableSection('what-it-means', '🎯 What it means', data.topic_meaning)}
                ${this.renderExpandableSection('why-it-matters', '💡 Why it matters', data.why_it_matters)}
                ${this.renderExpandableSection('real-life-analogy', '🔍 Real-life analogy', data.real_life_analogy)}
                ${this.renderInteractiveList('key-concepts', '🧩 Key concepts', data.key_concepts, true)}
                ${this.renderStepByStep(data.step_by_step_explanation)}
                ${this.renderInteractiveList('misconceptions', '⚠️ Common misconceptions', data.common_misconceptions)}
                ${this.renderInteractiveList('summary', '📋 Summary', data.summary)}
                ${this.renderNextTopics(data.suggested_next_topics)}
                ${data.learning_tips ? this.renderInteractiveList('learning-tips', '💡 Learning tips', data.learning_tips) : ''}
                
                <div class="breakdown-actions">
                    <button class="action-btn" onclick="thinkWiseApp.copyBreakdownToClipboard()">
                        📋 Copy to Clipboard
                    </button>
                    <button class="action-btn" onclick="thinkWiseApp.generateQuizFromBreakdown()">
                        🧠 Generate Quiz
                    </button>
                    <button class="action-btn" onclick="thinkWiseApp.saveBreakdownToNotes()">
                        💾 Save to Notes
                    </button>
                    <button class="action-btn" onclick="thinkWiseApp.shareBreakdown()">
                        🔗 Share Breakdown
                    </button>
                </div>
            </div>
        `;
    }

    renderExpandableSection(id, title, content) {
        if (!content) return '';
        
        return `
            <section class="concept-section expandable" id="${id}">
                <div class="section-header" onclick="thinkWiseApp.toggleSection('${id}')" role="button" aria-expanded="true" tabindex="0">
                    <h3 class="section-title">${title}</h3>
                    <span class="expand-icon">▼</span>
                </div>
                <div class="section-content">
                    <div class="content-text">${this.markdownToHtml(this.escapeHtml(content))}</div>
                    <div class="section-actions">
                        <button class="text-btn" onclick="thinkWiseApp.speakText('${id}')">
                            🔊 Read Aloud
                        </button>
                        <button class="text-btn" onclick="thinkWiseApp.simplifySection('${id}')">
                            🔄 Simplify
                        </button>
                    </div>
                </div>
            </section>
        `;
    }

    renderInteractiveList(id, title, items, isChecklist = false) {
        if (!items || !Array.isArray(items) || items.length === 0) return '';
        
        const itemsHtml = items.map((item, index) => `
            <li class="interactive-item" data-item-id="${id}-${index}">
                <div class="item-content">
                    ${isChecklist ? '<input type="checkbox" class="item-checkbox">' : ''}
                    <span class="item-text">${this.markdownToHtml(this.escapeHtml(item))}</span>
                </div>
                <div class="item-actions">
                    <button class="icon-btn" onclick="thinkWiseApp.explainItem('${id}-${index}')" title="Explain in detail">
                        💬
                    </button>
                    <button class="icon-btn" onclick="thinkWiseApp.addToFlashcards('${id}-${index}')" title="Add to flashcards">
                        📚
                    </button>
                </div>
            </li>
        `).join('');
        
        return `
            <section class="concept-section interactive-list" id="${id}">
                <h3 class="section-title">${title}</h3>
                <ul class="key-points interactive">${itemsHtml}</ul>
            </section>
        `;
    }

    renderStepByStep(steps) {
        if (!steps) return '';
        
        let stepsHtml = '';
        if (Array.isArray(steps)) {
            stepsHtml = steps.map((step, index) => `
                <li class="step-item" data-step="${index + 1}">
                    <div class="step-number">${index + 1}</div>
                    <div class="step-content">${this.markdownToHtml(this.escapeHtml(step))}</div>
                    <div class="step-actions">
                        <button class="icon-btn" onclick="thinkWiseApp.markStepComplete(${index + 1})" title="Mark as complete">
                            ✅
                        </button>
                    </div>
                </li>
            `).join('');
        } else {
            stepsHtml = `<div class="step-content">${this.markdownToHtml(this.escapeHtml(steps))}</div>`;
        }
        
        return `
            <section class="concept-section step-by-step" id="step-by-step">
                <h3 class="section-title">🚀 Step-by-step explanation</h3>
                <div class="steps-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 0%"></div>
                    </div>
                    <span class="progress-text">0/${Array.isArray(steps) ? steps.length : 1} steps completed</span>
                </div>
                ${Array.isArray(steps) ? `<ol class="steps-list">${stepsHtml}</ol>` : `<div class="single-step">${stepsHtml}</div>`}
            </section>
        `;
    }

    renderNextTopics(topics) {
        if (!topics || !Array.isArray(topics) || topics.length === 0) return '';
        
        const topicsHtml = topics.map(topic => `
            <div class="topic-chip" onclick="thinkWiseApp.quickBreakdown('${this.escapeHtml(topic)}')">
                ${this.escapeHtml(topic)}
            </div>
        `).join('');
        
        return `
            <section class="concept-section next-topics">
                <h3 class="section-title">🎯 Suggested next topics</h3>
                <div class="topics-grid">${topicsHtml}</div>
            </section>
        `;
    }

    setupInteractiveElements() {
        // Add event listeners for interactive elements
        const checkboxes = this.elements.resultsContainer.querySelectorAll('.item-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const item = e.target.closest('.interactive-item');
                if (e.target.checked) {
                    item.classList.add('completed');
                } else {
                    item.classList.remove('completed');
                }
            });
        });
    }

    // Utility Methods
    setLoadingState(isLoading) {
        if (isLoading) {
            this.elements.submitButton.disabled = true;
            this.elements.spinner.classList.remove("hidden");
            this.elements.submitButton.classList.add("loading");
            this.elements.submitButton.setAttribute("aria-busy", "true");
            
            // Update button text
            const buttonText = this.elements.submitButton.querySelector(".button-text");
            if (buttonText) {
                buttonText.textContent = "Analyzing with AI...";
            }
        } else {
            this.elements.submitButton.disabled = false;
            this.elements.spinner.classList.add("hidden");
            this.elements.submitButton.classList.remove("loading");
            this.elements.submitButton.removeAttribute("aria-busy");
            
            // Reset button text
            const buttonText = this.elements.submitButton.querySelector(".button-text");
            if (buttonText) {
                buttonText.textContent = "Break It Down";
            }
        }
    }

    escapeHtml(str) {
        if (!str) return "";
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    markdownToHtml(markdown) {
        if (!markdown) return '';
        
        return markdown
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    validateTopicInput() {
        const topic = this.elements.topicInput.value.trim();
        
        if (!topic) {
            this.showNotification("Please enter a topic to break down", "warning");
            this.elements.topicInput.focus();
            return false;
        }
        
        if (topic.length < 2) {
            this.showNotification("Topic must be at least 2 characters long", "warning");
            this.elements.topicInput.focus();
            return false;
        }
        
        if (topic.length > 500) {
            this.showNotification("Topic is too long. Please be more specific.", "warning");
            this.elements.topicInput.focus();
            return false;
        }
        
        return true;
    }

    showNotification(message, type = "info", duration = 5000) {
        // Remove existing notifications
        const existing = document.querySelector('.thinkwise-notification');
        if (existing) existing.remove();
        
        const notification = document.createElement('div');
        notification.className = `thinkwise-notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${this.getNotificationIcon(type)}</span>
                <span class="notification-message">${this.escapeHtml(message)}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, duration);
        }
    }

    getNotificationIcon(type) {
        const icons = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };
        return icons[type] || icons.info;
    }

    // Cache Management
    generateCacheKey(topic, complexity, learningStyle) {
        return `breakdown:${topic}:${complexity}:${learningStyle}`.toLowerCase();
    }

    getCachedResult(key) {
        try {
            const cached = localStorage.getItem(`thinkwise_${key}`);
            if (cached) {
                const data = JSON.parse(cached);
                // Check if cache is still valid (24 hours)
                if (Date.now() - data.timestamp < 24 * 60 * 60 * 1000) {
                    return data;
                }
            }
        } catch (e) {
            console.warn('Cache read error:', e);
        }
        return null;
    }

    cacheResult(key, data) {
        try {
            const cacheData = {
                ...data,
                timestamp: Date.now()
            };
            localStorage.setItem(`thinkwise_${key}`, JSON.stringify(cacheData));
        } catch (e) {
            console.warn('Cache write error:', e);
        }
    }

    // Session Management
    startSessionTimer() {
        this.state.sessionStartTime = Date.now();
        
        setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.state.sessionStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            
            if (this.elements.sessionTimer) {
                this.elements.sessionTimer.textContent = 
                    `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }

    generateRequestId() {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    getSessionId() {
        let sessionId = localStorage.getItem('thinkwise_session_id');
        if (!sessionId) {
            sessionId = `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            localStorage.setItem('thinkwise_session_id', sessionId);
        }
        return sessionId;
    }

    // History Management
    loadBreakdownHistory() {
        try {
            const stored = localStorage.getItem('thinkwise_breakdown_history');
            this.state.breakdownHistory = stored ? JSON.parse(stored) : [];
        } catch (error) {
            console.error('Failed to load history:', error);
            this.state.breakdownHistory = [];
        }
    }

    saveToHistory(data, metadata) {
        const historyItem = {
            id: this.generateRequestId(),
            topic: metadata.topic,
            complexity: metadata.complexity,
            timestamp: Date.now(),
            data: data
        };
        
        this.state.breakdownHistory.unshift(historyItem);
        
        // Keep only last 50 items
        if (this.state.breakdownHistory.length > 50) {
            this.state.breakdownHistory = this.state.breakdownHistory.slice(0, 50);
        }
        
        try {
            localStorage.setItem('thinkwise_breakdown_history', JSON.stringify(this.state.breakdownHistory));
        } catch (error) {
            console.error('Failed to save history:', error);
        }
    }

    // Interactive Methods (called from HTML onclick)
    toggleSection(sectionId) {
        const section = document.getElementById(sectionId);
        const content = section.querySelector('.section-content');
        const header = section.querySelector('.section-header');
        const icon = section.querySelector('.expand-icon');
        
        const isExpanded = header.getAttribute('aria-expanded') === 'true';
        
        if (isExpanded) {
            content.style.display = 'none';
            header.setAttribute('aria-expanded', 'false');
            icon.textContent = '▶';
        } else {
            content.style.display = 'block';
            header.setAttribute('aria-expanded', 'true');
            icon.textContent = '▼';
        }
    }

    async copyBreakdownToClipboard() {
        const breakdown = document.querySelector('.enhanced-breakdown');
        if (!breakdown) {
            this.showNotification('No breakdown to copy', 'warning');
            return;
        }
        
        const textContent = breakdown.innerText;
        
        try {
            await navigator.clipboard.writeText(textContent);
            this.showNotification('Breakdown copied to clipboard!', 'success');
        } catch (error) {
            console.error('Failed to copy:', error);
            this.showNotification('Failed to copy to clipboard', 'error');
        }
    }

    quickBreakdown(topic) {
        this.elements.topicInput.value = topic;
        this.elements.form.dispatchEvent(new Event('submit'));
    }

    // Animation Methods
    animateResultsAppearance() {
        const sections = this.elements.resultsContainer.querySelectorAll('.concept-section');
        sections.forEach((section, index) => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                section.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                section.style.opacity = '1';
                section.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }

    // Placeholder methods for future implementation
    toggleVoiceInput() {
        if (this.state.voiceRecognition) {
            if (this.state.isListening) {
                this.state.voiceRecognition.stop();
            } else {
                this.state.voiceRecognition.start();
            }
        }
    }

    toggleAssistant() {
        this.state.isAssistantOpen = !this.state.isAssistantOpen;
        if (this.elements.assistantWindow) {
            this.elements.assistantWindow.classList.toggle('hidden');
        }
    }

    sendAssistantMessage() {
        // Implementation for AI assistant
        this.showNotification('AI assistant coming soon!', 'info');
    }

    exportBreakdown() {
        this.showNotification('Export feature coming soon!', 'info');
    }

    generateQuiz() {
        this.showNotification('Quiz generation coming soon!', 'info');
    }

    saveToNotes() {
        this.showNotification('Notes saving coming soon!', 'info');
    }

    shareBreakdown() {
        this.showNotification('Share feature coming soon!', 'info');
    }

    speakText(sectionId) {
        this.showNotification('Text-to-speech coming soon!', 'info');
    }

    simplifySection(sectionId) {
        this.showNotification('Section simplification coming soon!', 'info');
    }

    explainItem(itemId) {
        this.showNotification('Item explanation coming soon!', 'info');
    }

    addToFlashcards(itemId) {
        this.showNotification('Flashcards feature coming soon!', 'info');
    }

    markStepComplete(stepNumber) {
        this.showNotification(`Step ${stepNumber} marked complete!`, 'success');
    }

    fetchSuggestions(query) {
        // Implementation for AI-powered suggestions
        const mockSuggestions = [
            `${query} basics`,
            `Advanced ${query}`,
            `${query} applications`,
            `${query} theory`
        ];
        
        this.showSuggestions(mockSuggestions);
    }

    showSuggestions(suggestions) {
        if (!this.elements.suggestionsPanel) return;
        
        const html = suggestions.map(suggestion => `
            <div class="suggestion-item" onclick="thinkWiseApp.applySuggestion('${this.escapeHtml(suggestion)}')">
                ${this.escapeHtml(suggestion)}
            </div>
        `).join('');
        
        this.elements.suggestionsPanel.innerHTML = html;
        this.elements.suggestionsPanel.classList.remove('hidden');
    }

    applySuggestion(suggestion) {
        this.elements.topicInput.value = suggestion;
        this.hideSuggestions();
        this.elements.topicInput.focus();
    }

    hideSuggestions() {
        if (this.elements.suggestionsPanel) {
            this.elements.suggestionsPanel.classList.add('hidden');
        }
    }

    updateRelatedSuggestions(topics) {
        // Update UI with related topics
        console.log('Related topics:', topics);
    }

    trackUserActivity(type, topic, complexity) {
        // Analytics tracking
        console.log(`Tracked: ${type} for ${topic} at ${complexity} level`);
    }

    loadUserPreferences() {
        try {
            this.state.userPreferences = JSON.parse(localStorage.getItem('thinkwise_preferences') || '{}');
        } catch (error) {
            this.state.userPreferences = {};
        }
    }
}

// Initialize the application
let thinkWiseApp;

document.addEventListener("DOMContentLoaded", () => {
    thinkWiseApp = new ThinkWiseApp();
    
    // Make app globally available for interactive elements
    window.thinkWiseApp = thinkWiseApp;
});

// Export for module usage (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThinkWiseApp;
}