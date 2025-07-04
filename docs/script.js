/**
 * RSS Article Analysis Dashboard
 * Professional JavaScript Architecture
 * 
 * @author RSS Analyzer Team
 * @version 2.0.0
 */

'use strict';

/**
 * Application Configuration
 */
const CONFIG = {
    DATA_URL: 'data.json',
    DEBOUNCE_DELAY: 300,
    ANIMATION_DURATION: 250,
    STORAGE_KEY: 'rss-analyzer-preferences',
    DATE_FORMAT_OPTIONS: {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    }
};

/**
 * Application State Management
 */
class AppState {
    constructor() {
        this.data = {
            articles: [],
            groupedArticles: {},
            filteredArticles: [],
            isLoading: true,
            hasError: false,
            lastUpdated: null,
            processingStatus: null
        };
        
        this.filters = {
            searchTerm: ''
        };
        
        this.ui = {
            expandedSections: new Set(),
            preferences: this.loadPreferences()
        };
    }
    
    /**
     * Load user preferences from localStorage
     * @returns {Object} User preferences
     */
    loadPreferences() {
        try {
            const stored = localStorage.getItem(CONFIG.STORAGE_KEY);
            return stored ? JSON.parse(stored) : { expandedSections: [] };
        } catch (error) {
            console.warn('Failed to load preferences:', error);
            return { expandedSections: [] };
        }
    }
    
    /**
     * Save user preferences to localStorage
     */
    savePreferences() {
        try {
            const preferences = {
                expandedSections: Array.from(this.ui.expandedSections)
            };
            localStorage.setItem(CONFIG.STORAGE_KEY, JSON.stringify(preferences));
        } catch (error) {
            console.warn('Failed to save preferences:', error);
        }
    }
    
    /**
     * Update application state
     * @param {Object} updates - State updates
     */
    update(updates) {
        Object.assign(this.data, updates);
        this.render();
    }
    
    /**
     * Update filters and trigger re-render
     * @param {Object} filterUpdates - Filter updates
     */
    updateFilters(filterUpdates) {
        Object.assign(this.filters, filterUpdates);
        this.filterAndRender();
    }
    
    /**
     * Filter articles based on current filters
     */
    filterAndRender() {
        this.data.filteredArticles = this.data.articles.filter(article => {
            return this.matchesSearch(article);
        });
        
        this.data.groupedArticles = this.groupArticlesByDate(this.data.filteredArticles);
        this.render();
    }
    
    /**
     * Check if article matches search term
     * @param {Object} article - Article to check
     * @returns {boolean} Whether article matches search
     */
    matchesSearch(article) {
        if (!this.filters.searchTerm) return true;
        
        const searchTerm = this.filters.searchTerm.toLowerCase();
        const searchableFields = [
            article.title,
            article.analysis,
            ...(article.linked_articles?.map(la => la.title) || [])
        ];
        
        return searchableFields.some(field => 
            field?.toLowerCase().includes(searchTerm)
        );
    }
    
    
    /**
     * Group articles by date
     * @param {Array} articles - Articles to group
     * @returns {Object} Grouped articles
     */
    groupArticlesByDate(articles) {
        const grouped = {};
        const today = new Date().toDateString();
        
        articles.forEach(article => {
            const date = new Date(article.processed_date);
            const dateString = date.toDateString();
            const isToday = dateString === today;
            
            const dateKey = isToday ? 'Today' : Utils.formatDate(date);
            
            if (!grouped[dateKey]) {
                grouped[dateKey] = {
                    articles: [],
                    isToday,
                    date,
                    count: 0
                };
            }
            
            grouped[dateKey].articles.push(article);
            grouped[dateKey].count++;
        });
        
        // Sort groups by date (newest first)
        return Object.keys(grouped)
            .sort((a, b) => grouped[b].date - grouped[a].date)
            .reduce((acc, key) => {
                acc[key] = grouped[key];
                return acc;
            }, {});
    }
    
    /**
     * Render the application
     */
    render() {
        if (this.data.isLoading) {
            this.showLoading();
        } else if (this.data.hasError) {
            this.showError();
        } else if (this.data.filteredArticles.length === 0) {
            this.showEmpty();
        } else {
            this.showArticles();
        }
        
        this.updateStats();
        this.updateLastUpdated();
    }
    
    /**
     * Show loading state
     */
    showLoading() {
        UI.showElement('loading-container');
        UI.hideElement('error-container');
        UI.hideElement('empty-container');
        UI.hideElement('articles-container');
    }
    
    /**
     * Show error state
     */
    showError() {
        UI.hideElement('loading-container');
        UI.showElement('error-container');
        UI.hideElement('empty-container');
        UI.hideElement('articles-container');
    }
    
    /**
     * Show empty state
     */
    showEmpty() {
        UI.hideElement('loading-container');
        UI.hideElement('error-container');
        UI.showElement('empty-container');
        UI.hideElement('articles-container');
    }
    
    /**
     * Show articles
     */
    showArticles() {
        UI.hideElement('loading-container');
        UI.hideElement('error-container');
        UI.hideElement('empty-container');
        UI.showElement('articles-container');
        
        this.renderArticleGroups();
    }
    
    /**
     * Render article groups
     */
    renderArticleGroups() {
        const container = document.getElementById('articles-container');
        container.innerHTML = '';
        
        Object.entries(this.data.groupedArticles).forEach(([dateKey, group]) => {
            const dateGroupElement = this.createDateGroup(dateKey, group);
            container.appendChild(dateGroupElement);
        });
    }
    
    /**
     * Create date group element
     * @param {string} dateKey - Date key
     * @param {Object} group - Article group
     * @returns {HTMLElement} Date group element
     */
    createDateGroup(dateKey, group) {
        const dateGroup = document.createElement('div');
        dateGroup.className = 'date-group';
        dateGroup.setAttribute('data-date', dateKey);
        
        const isExpanded = group.isToday || this.ui.expandedSections.has(dateKey);
        
        dateGroup.innerHTML = `
            <div class="date-header ${isExpanded ? 'expanded' : ''}" 
                 role="button" 
                 tabindex="0" 
                 aria-expanded="${isExpanded}"
                 aria-controls="articles-${Utils.sanitizeId(dateKey)}">
                <h3 class="date-title">${dateKey}</h3>
                <div class="date-meta">
                    <span class="date-count">${group.count}</span>
                    <span class="expand-icon" aria-hidden="true">▼</span>
                </div>
            </div>
            <div id="articles-${Utils.sanitizeId(dateKey)}" 
                 class="articles-list ${isExpanded ? 'expanded' : ''}"
                 aria-hidden="${!isExpanded}">
                ${group.articles.map(article => this.createArticleCard(article)).join('')}
            </div>
        `;
        
        // Add click event listener
        const header = dateGroup.querySelector('.date-header');
        header.addEventListener('click', () => this.toggleSection(dateKey));
        header.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.toggleSection(dateKey);
            }
        });
        
        return dateGroup;
    }
    
    /**
     * Create article card HTML
     * @param {Object} article - Article data
     * @returns {string} Article card HTML
     */
    createArticleCard(article) {
        const processedDate = Utils.formatDateTime(article.processed_date);
        
        const linkedArticlesHtml = article.linked_articles?.length > 0 ? `
            <div class="linked-articles">
                <h4 class="linked-articles-title">Referenced Articles (${article.linked_articles.length})</h4>
                ${article.linked_articles.map(linked => `
                    <div class="linked-article">
                        <a href="${Utils.escapeHtml(linked.url)}" 
                           target="_blank" 
                           rel="noopener noreferrer">
                            ${Utils.escapeHtml(linked.title)}
                        </a>
                    </div>
                `).join('')}
            </div>
        ` : '';
        
        return `
            <article class="article-card">
                <header class="article-header">
                    <h4 class="article-title">
                        <a href="${Utils.escapeHtml(article.url)}" 
                           target="_blank" 
                           rel="noopener noreferrer">
                            ${Utils.escapeHtml(article.title)}
                        </a>
                    </h4>
                    <div class="article-meta">
                        <time datetime="${article.processed_date}">
                            Processed: ${processedDate}
                        </time>
                    </div>
                </header>
                <div class="article-analysis">
                    ${Utils.formatAnalysis(article.analysis)}
                </div>
                ${linkedArticlesHtml}
            </article>
        `;
    }
    
    /**
     * Toggle section expanded state
     * @param {string} dateKey - Date key to toggle
     */
    toggleSection(dateKey) {
        const dateGroup = document.querySelector(`[data-date="${dateKey}"]`);
        if (!dateGroup) return;
        
        const header = dateGroup.querySelector('.date-header');
        const list = dateGroup.querySelector('.articles-list');
        const isExpanded = header.classList.contains('expanded');
        
        if (isExpanded) {
            header.classList.remove('expanded');
            list.classList.remove('expanded');
            header.setAttribute('aria-expanded', 'false');
            list.setAttribute('aria-hidden', 'true');
            this.ui.expandedSections.delete(dateKey);
        } else {
            header.classList.add('expanded');
            list.classList.add('expanded');
            header.setAttribute('aria-expanded', 'true');
            list.setAttribute('aria-hidden', 'false');
            this.ui.expandedSections.add(dateKey);
        }
        
        this.savePreferences();
    }
    
    /**
     * Update statistics display
     */
    updateStats() {
        const today = new Date().toDateString();
        const todayArticles = this.data.articles.filter(article => 
            new Date(article.processed_date).toDateString() === today
        ).length;
        
        document.getElementById('total-articles').textContent = this.data.articles.length;
        document.getElementById('today-articles').textContent = todayArticles;
        
        // Get AI provider from first article
        if (this.data.articles.length > 0 && this.data.articles[0].ai_provider) {
            document.getElementById('ai-provider').textContent = 
                this.data.articles[0].ai_provider.charAt(0).toUpperCase() + 
                this.data.articles[0].ai_provider.slice(1);
        }
        
        // Update system status
        this.updateSystemStatus();
    }
    
    /**
     * Update system status display
     */
    updateSystemStatus() {
        const statusElement = document.getElementById('system-status');
        if (!statusElement) return;
        
        const status = this.data.processingStatus;
        if (!status) {
            statusElement.textContent = 'Unknown';
            statusElement.className = 'stat-value system-status status-unknown';
            return;
        }
        
        const systemStatus = status.system_status || 'unknown';
        
        // Update status text and class
        statusElement.className = `stat-value system-status status-${systemStatus}`;
        
        switch (systemStatus) {
            case 'success':
                statusElement.textContent = 'Healthy';
                statusElement.title = 'All systems operational';
                break;
            case 'partial':
                statusElement.textContent = 'Partial';
                statusElement.title = 'Some issues detected';
                break;
            case 'failed':
                statusElement.textContent = 'Issues';
                statusElement.title = 'System experiencing problems';
                this.showRecentErrors(status);
                break;
            default:
                statusElement.textContent = 'Unknown';
                statusElement.title = 'Status unknown';
        }
    }
    
    /**
     * Show recent errors as alerts
     */
    showRecentErrors(status) {
        const recentErrors = status.recent_errors_by_date || {};
        const today = new Date().toISOString().split('T')[0];
        const todayErrors = recentErrors[today] || [];
        
        if (todayErrors.length > 0) {
            // Find a good place to show errors (after the stats section)
            const statsSection = document.querySelector('.stats-section');
            
            // Remove any existing error alerts
            const existingAlerts = document.querySelectorAll('.error-alert');
            existingAlerts.forEach(alert => alert.remove());
            
            // Create error alert
            const errorAlert = document.createElement('div');
            errorAlert.className = 'error-alert';
            errorAlert.innerHTML = `
                <h4>⚠️ Processing Issues Detected</h4>
                <p>There were ${todayErrors.length} error(s) during today's processing:</p>
                <div class="error-details">
                    ${todayErrors.map(error => `
                        <div>
                            <strong>${error.component}:</strong> ${error.message}
                            <div class="error-timestamp">${Utils.formatDateTime(error.timestamp)}</div>
                        </div>
                    `).join('')}
                </div>
            `;
            
            // Insert after stats section
            if (statsSection && statsSection.nextSibling) {
                statsSection.parentNode.insertBefore(errorAlert, statsSection.nextSibling);
            }
        }
    }
    
    /**
     * Update last updated timestamp
     */
    updateLastUpdated() {
        if (this.data.lastUpdated) {
            const formatted = Utils.formatDateTime(this.data.lastUpdated);
            document.getElementById('last-updated').textContent = `Last updated: ${formatted}`;
        }
    }
}

/**
 * Utility Functions
 */
class Utils {
    
    /**
     * Format date for display
     * @param {Date} date - Date to format
     * @returns {string} Formatted date
     */
    static formatDate(date) {
        return date.toLocaleDateString('en-US', CONFIG.DATE_FORMAT_OPTIONS);
    }
    
    /**
     * Format date and time for display
     * @param {string} dateString - ISO date string
     * @returns {string} Formatted date and time
     */
    static formatDateTime(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    /**
     * Escape HTML to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    /**
     * Sanitize string for use as ID
     * @param {string} str - String to sanitize
     * @returns {string} Sanitized string
     */
    static sanitizeId(str) {
        return str.replace(/[^a-zA-Z0-9-_]/g, '-').toLowerCase();
    }
    
    /**
     * Format analysis text with proper HTML
     * @param {string} analysis - Analysis text
     * @returns {string} Formatted HTML
     */
    static formatAnalysis(analysis) {
        return analysis
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>');
    }
    
    /**
     * Debounce function calls
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in milliseconds
     * @returns {Function} Debounced function
     */
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    /**
     * Show loading toast notification
     * @param {string} message - Message to show
     */
    static showToast(message) {
        // Simple toast implementation
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--color-primary);
            color: var(--color-white);
            padding: 12px 24px;
            border-radius: 8px;
            z-index: 10000;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;
        
        document.body.appendChild(toast);
        
        // Fade in
        setTimeout(() => toast.style.opacity = '1', 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => document.body.removeChild(toast), 300);
        }, 3000);
    }
}

/**
 * UI Helper Functions
 */
class UI {
    /**
     * Show element by ID
     * @param {string} id - Element ID
     */
    static showElement(id) {
        const element = document.getElementById(id);
        if (element) element.style.display = 'block';
    }
    
    /**
     * Hide element by ID
     * @param {string} id - Element ID
     */
    static hideElement(id) {
        const element = document.getElementById(id);
        if (element) element.style.display = 'none';
    }
    
    /**
     * Toggle element visibility
     * @param {string} id - Element ID
     */
    static toggleElement(id) {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = element.style.display === 'none' ? 'block' : 'none';
        }
    }
}

/**
 * Data Service
 */
class DataService {
    /**
     * Fetch articles data
     * @returns {Promise<Object>} Articles data
     */
    static async fetchArticles() {
        try {
            const response = await fetch(CONFIG.DATA_URL);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Validate data structure
            if (!data.articles || !Array.isArray(data.articles)) {
                throw new Error('Invalid data format: missing articles array');
            }
            
            return data;
        } catch (error) {
            console.error('Failed to fetch articles:', error);
            throw error;
        }
    }
    
    /**
     * Retry fetch with exponential backoff
     * @param {number} maxRetries - Maximum number of retries
     * @returns {Promise<Object>} Articles data
     */
    static async fetchWithRetry(maxRetries = 3) {
        let lastError;
        
        for (let i = 0; i < maxRetries; i++) {
            try {
                return await this.fetchArticles();
            } catch (error) {
                lastError = error;
                if (i < maxRetries - 1) {
                    const delay = Math.pow(2, i) * 1000; // Exponential backoff
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }
        
        throw lastError;
    }
}

/**
 * Event Handlers
 */
class EventHandlers {
    constructor(appState) {
        this.appState = appState;
        this.setupEventListeners();
    }
    
    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        this.setupSearchHandler();
        this.setupControlButtons();
        this.setupRetryButton();
        this.setupKeyboardNavigation();
    }
    
    /**
     * Setup search input handler
     */
    setupSearchHandler() {
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            const debouncedSearch = Utils.debounce((value) => {
                this.appState.updateFilters({ searchTerm: value });
            }, CONFIG.DEBOUNCE_DELAY);
            
            searchInput.addEventListener('input', (e) => {
                debouncedSearch(e.target.value);
            });
        }
    }
    
    
    /**
     * Setup control button handlers
     */
    setupControlButtons() {
        const expandAllBtn = document.getElementById('expand-all');
        const collapseAllBtn = document.getElementById('collapse-all');
        
        if (expandAllBtn) {
            expandAllBtn.addEventListener('click', () => {
                this.expandAllSections();
            });
        }
        
        if (collapseAllBtn) {
            collapseAllBtn.addEventListener('click', () => {
                this.collapseAllSections();
            });
        }
    }
    
    /**
     * Setup retry button handler
     */
    setupRetryButton() {
        const retryBtn = document.getElementById('retry-btn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => {
                this.retryLoad();
            });
        }
    }
    
    /**
     * Setup keyboard navigation
     */
    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.getElementById('search-input');
                if (searchInput) {
                    searchInput.focus();
                    searchInput.select();
                }
            }
        });
    }
    
    /**
     * Expand all sections
     */
    expandAllSections() {
        const headers = document.querySelectorAll('.date-header');
        headers.forEach(header => {
            const dateKey = header.closest('.date-group').dataset.date;
            if (!header.classList.contains('expanded')) {
                this.appState.toggleSection(dateKey);
            }
        });
        Utils.showToast('All sections expanded');
    }
    
    /**
     * Collapse all sections
     */
    collapseAllSections() {
        const headers = document.querySelectorAll('.date-header');
        headers.forEach(header => {
            const dateKey = header.closest('.date-group').dataset.date;
            if (header.classList.contains('expanded')) {
                this.appState.toggleSection(dateKey);
            }
        });
        Utils.showToast('All sections collapsed');
    }
    
    /**
     * Retry loading data
     */
    async retryLoad() {
        this.appState.update({ isLoading: true, hasError: false });
        
        try {
            const data = await DataService.fetchWithRetry();
            this.appState.update({
                articles: data.articles,
                lastUpdated: data.generated_at,
                processingStatus: data.processing_status,
                isLoading: false,
                hasError: false
            });
            this.appState.filterAndRender();
            Utils.showToast('Articles loaded successfully');
        } catch (error) {
            this.appState.update({ isLoading: false, hasError: true });
            Utils.showToast('Failed to load articles');
        }
    }
}

/**
 * Application Controller
 */
class App {
    constructor() {
        this.state = new AppState();
        this.eventHandlers = new EventHandlers(this.state);
    }
    
    /**
     * Initialize the application
     */
    async init() {
        try {
            this.state.update({ isLoading: true, hasError: false });
            
            const data = await DataService.fetchWithRetry();
            
            this.state.update({
                articles: data.articles,
                lastUpdated: data.generated_at,
                isLoading: false,
                hasError: false
            });
            
            // Initialize UI state from preferences
            this.state.ui.expandedSections = new Set(this.state.ui.preferences.expandedSections);
            
            this.state.filterAndRender();
            
        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.state.update({ isLoading: false, hasError: true });
        }
    }
}

/**
 * Application Entry Point
 */
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.init();
    
    // Make app globally available for debugging
    if (typeof window !== 'undefined') {
        window.RSSAnalyzer = app;
    }
});