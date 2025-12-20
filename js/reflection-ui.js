/**
 * MYND Reflection UI
 * ==================
 * UI components for the Reflection Daemon:
 *   - Settings panel integration
 *   - Reflection Queue viewer
 *   - Notification badge
 *   - Activity log viewer
 */

const ReflectionUI = {
    // ═══════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════

    initialized: false,
    panelVisible: false,
    pendingCount: 0,

    // Store bound event listener references for cleanup
    _boundListeners: {
        reflectionComplete: null,
        itemApproved: null,
        itemDismissed: null,
        started: null,
        stopped: null,
        reflectionStarted: null,
        escapeHandler: null
    },

    // ═══════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════

    init() {
        if (this.initialized) return this;

        console.log('🔮 ReflectionUI: Initializing...');

        // Setup event listeners
        this.setupEventListeners();

        // Initial badge update
        this.updateBadge();

        this.initialized = true;
        console.log('✅ ReflectionUI: Ready');

        return this;
    },

    setupEventListeners() {
        // Clean up any existing listeners first
        this.cleanupEventListeners();

        // Create bound listener references for cleanup
        this._boundListeners.reflectionComplete = (e) => {
            this.updateBadge();
            this.showNotification(e.detail);
        };

        this._boundListeners.itemApproved = () => {
            this.updateBadge();
            this.refreshQueueView();
        };

        this._boundListeners.itemDismissed = () => {
            this.updateBadge();
            this.refreshQueueView();
        };

        this._boundListeners.started = () => {
            this.updateStatusIndicator(true);
        };

        this._boundListeners.stopped = () => {
            this.updateStatusIndicator(false);
        };

        this._boundListeners.reflectionStarted = () => {
            this.showReflectionInProgress();
        };

        // Add event listeners with stored references
        document.addEventListener('reflection:reflectionComplete', this._boundListeners.reflectionComplete);
        document.addEventListener('reflection:itemApproved', this._boundListeners.itemApproved);
        document.addEventListener('reflection:itemDismissed', this._boundListeners.itemDismissed);
        document.addEventListener('reflection:started', this._boundListeners.started);
        document.addEventListener('reflection:stopped', this._boundListeners.stopped);
        document.addEventListener('reflection:reflectionStarted', this._boundListeners.reflectionStarted);
    },

    cleanupEventListeners() {
        // Remove all stored event listeners to prevent memory leaks
        if (this._boundListeners.reflectionComplete) {
            document.removeEventListener('reflection:reflectionComplete', this._boundListeners.reflectionComplete);
        }
        if (this._boundListeners.itemApproved) {
            document.removeEventListener('reflection:itemApproved', this._boundListeners.itemApproved);
        }
        if (this._boundListeners.itemDismissed) {
            document.removeEventListener('reflection:itemDismissed', this._boundListeners.itemDismissed);
        }
        if (this._boundListeners.started) {
            document.removeEventListener('reflection:started', this._boundListeners.started);
        }
        if (this._boundListeners.stopped) {
            document.removeEventListener('reflection:stopped', this._boundListeners.stopped);
        }
        if (this._boundListeners.reflectionStarted) {
            document.removeEventListener('reflection:reflectionStarted', this._boundListeners.reflectionStarted);
        }
        if (this._boundListeners.escapeHandler) {
            document.removeEventListener('keydown', this._boundListeners.escapeHandler);
        }
    },

    // ═══════════════════════════════════════════════════════════════════
    // NOTIFICATION BADGE
    // ═══════════════════════════════════════════════════════════════════

    async updateBadge() {
        if (typeof ReflectionDaemon === 'undefined') return;

        const count = await ReflectionDaemon.getPendingCount();
        this.pendingCount = count;

        // Update badge on Neural panel icon
        const badge = document.getElementById('reflection-badge');
        if (badge) {
            if (count > 0) {
                badge.textContent = count > 99 ? '99+' : count;
                badge.style.display = 'flex';
            } else {
                badge.style.display = 'none';
            }
        }

        // Also update any other badge locations
        const headerBadge = document.getElementById('reflection-header-badge');
        if (headerBadge) {
            if (count > 0) {
                headerBadge.textContent = count > 99 ? '99+' : count;
                headerBadge.style.display = 'flex';
            } else {
                headerBadge.style.display = 'none';
            }
        }
    },

    showNotification(detail) {
        const total = (detail.insightCount || 0) +
                      (detail.improvementCount || 0) +
                      (detail.connectionCount || 0) +
                      (detail.codeIssueCount || 0);

        if (total === 0) return;

        // Use existing toast system if available
        if (typeof showToast === 'function') {
            showToast(`🔮 Reflection complete: ${total} new insights`, 4000);
        } else {
            console.log(`🔮 Reflection complete: ${total} new insights`);
        }
    },

    updateStatusIndicator(isRunning) {
        const indicator = document.getElementById('reflection-status-indicator');
        if (indicator) {
            indicator.className = isRunning ? 'reflection-status-active' : 'reflection-status-inactive';
            indicator.title = isRunning ? 'Autonomous mode active' : 'Autonomous mode inactive';
        }
    },

    showReflectionInProgress() {
        const indicator = document.getElementById('reflection-status-indicator');
        if (indicator) {
            indicator.classList.add('reflecting');
            setTimeout(() => {
                indicator.classList.remove('reflecting');
            }, 3000);
        }
    },

    // ═══════════════════════════════════════════════════════════════════
    // SETTINGS PANEL
    // ═══════════════════════════════════════════════════════════════════

    renderSettingsSection() {
        if (typeof ReflectionDaemon === 'undefined') {
            return '<div class="settings-section-disabled">Reflection Daemon not loaded</div>';
        }

        const status = ReflectionDaemon.getStatus();
        const stats = ReflectionDaemon.getStats();

        return `
            <div class="settings-section reflection-settings">
                <h3 class="settings-section-title">
                    <span class="settings-icon">🔮</span>
                    Autonomous Reflection
                </h3>

                <div class="settings-row">
                    <label class="settings-label">
                        <span>Enable Autonomous Mode</span>
                        <span class="settings-hint">AI reflects during idle periods</span>
                    </label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="reflection-toggle" ${status.enabled ? 'checked' : ''}>
                        <span class="toggle-slider"></span>
                    </label>
                </div>

                <div class="settings-row">
                    <label class="settings-label">
                        <span>Reflection Frequency</span>
                    </label>
                    <select id="reflection-frequency" class="settings-select">
                        <option value="15min" ${status.config.reflectionIntervalMs === 15*60*1000 ? 'selected' : ''}>Every 15 minutes</option>
                        <option value="30min" ${status.config.reflectionIntervalMs === 30*60*1000 ? 'selected' : ''}>Every 30 minutes</option>
                        <option value="1hr" ${status.config.reflectionIntervalMs === 60*60*1000 ? 'selected' : ''}>Every hour</option>
                        <option value="2hr" ${status.config.reflectionIntervalMs === 2*60*60*1000 ? 'selected' : ''}>Every 2 hours</option>
                    </select>
                </div>

                <div class="settings-row">
                    <label class="settings-label">
                        <span>Idle Threshold</span>
                        <span class="settings-hint">Time before reflection triggers</span>
                    </label>
                    <select id="reflection-idle-threshold" class="settings-select">
                        <option value="2" ${status.config.idleThresholdMs === 2*60*1000 ? 'selected' : ''}>2 minutes</option>
                        <option value="5" ${status.config.idleThresholdMs === 5*60*1000 ? 'selected' : ''}>5 minutes</option>
                        <option value="10" ${status.config.idleThresholdMs === 10*60*1000 ? 'selected' : ''}>10 minutes</option>
                        <option value="15" ${status.config.idleThresholdMs === 15*60*1000 ? 'selected' : ''}>15 minutes</option>
                    </select>
                </div>

                <div class="settings-row">
                    <label class="settings-label">
                        <span>Auto-add to Map</span>
                        <span class="settings-hint">Add reflection logs to MYND Thoughts</span>
                    </label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="reflection-auto-add" ${status.config.autoAddToMap ? 'checked' : ''}>
                        <span class="toggle-slider"></span>
                    </label>
                </div>

                <div class="reflection-stats">
                    <div class="stat-item">
                        <span class="stat-value">${stats.totalReflections}</span>
                        <span class="stat-label">Reflections</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${stats.insightsGenerated}</span>
                        <span class="stat-label">Insights</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${stats.approvedCount}</span>
                        <span class="stat-label">Approved</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${this.pendingCount}</span>
                        <span class="stat-label">Pending</span>
                    </div>
                </div>

                <div class="settings-actions">
                    <button id="reflection-view-queue" class="settings-btn secondary">
                        View Queue (${this.pendingCount})
                    </button>
                    <button id="reflection-trigger-now" class="settings-btn primary" ${!status.enabled ? 'disabled' : ''}>
                        Reflect Now
                    </button>
                </div>
            </div>
        `;
    },

    attachSettingsListeners() {
        // Toggle autonomous mode
        const toggle = document.getElementById('reflection-toggle');
        if (toggle) {
            toggle.addEventListener('change', () => {
                ReflectionDaemon.toggle();
            });
        }

        // Frequency select
        const frequency = document.getElementById('reflection-frequency');
        if (frequency) {
            frequency.addEventListener('change', () => {
                ReflectionDaemon.setFrequency(frequency.value);
            });
        }

        // Idle threshold select
        const idleThreshold = document.getElementById('reflection-idle-threshold');
        if (idleThreshold) {
            idleThreshold.addEventListener('change', () => {
                const minutes = parseInt(idleThreshold.value, 10);
                ReflectionDaemon.setConfig({ idleThresholdMs: minutes * 60 * 1000 });
            });
        }

        // Auto-add toggle
        const autoAdd = document.getElementById('reflection-auto-add');
        if (autoAdd) {
            autoAdd.addEventListener('change', () => {
                ReflectionDaemon.setConfig({ autoAddToMap: autoAdd.checked });
            });
        }

        // View queue button
        const viewQueue = document.getElementById('reflection-view-queue');
        if (viewQueue) {
            viewQueue.addEventListener('click', () => {
                this.showQueuePanel();
            });
        }

        // Trigger now button
        const triggerNow = document.getElementById('reflection-trigger-now');
        if (triggerNow) {
            triggerNow.addEventListener('click', async () => {
                triggerNow.disabled = true;
                triggerNow.textContent = 'Reflecting...';
                await ReflectionDaemon.triggerReflection('manual');
                triggerNow.disabled = false;
                triggerNow.textContent = 'Reflect Now';
                this.updateBadge();
            });
        }
    },

    // ═══════════════════════════════════════════════════════════════════
    // QUEUE PANEL
    // ═══════════════════════════════════════════════════════════════════

    async showQueuePanel() {
        // Create or show the panel
        let panel = document.getElementById('reflection-queue-panel');

        if (!panel) {
            panel = document.createElement('div');
            panel.id = 'reflection-queue-panel';
            panel.className = 'reflection-queue-panel';
            document.body.appendChild(panel);
        }

        // Render content
        panel.innerHTML = await this.renderQueueContent();
        panel.classList.add('visible');
        this.panelVisible = true;

        // Attach listeners
        this.attachQueueListeners();
    },

    hideQueuePanel() {
        const panel = document.getElementById('reflection-queue-panel');
        if (panel) {
            panel.classList.remove('visible');
            this.panelVisible = false;
        }
    },

    async renderQueueContent() {
        const items = await ReflectionDaemon.getQueue({ status: 'pending' });
        const log = ReflectionDaemon.getActivityLog().slice(0, 20);

        return `
            <div class="queue-header">
                <h2>🔮 Reflection Queue</h2>
                <button class="queue-close-btn" id="queue-close">×</button>
            </div>

            <div class="queue-tabs">
                <button class="queue-tab active" data-tab="pending">Pending (${items.length})</button>
                <button class="queue-tab" data-tab="log">Activity Log</button>
            </div>

            <div class="queue-content" id="queue-tab-pending">
                ${items.length === 0 ? `
                    <div class="queue-empty">
                        <div class="queue-empty-icon">🔮</div>
                        <p>No pending reflections</p>
                        <p class="queue-empty-hint">Reflections will appear here when the AI analyzes your map</p>
                    </div>
                ` : `
                    <div class="queue-items">
                        ${items.map(item => this.renderQueueItem(item)).join('')}
                    </div>
                `}
            </div>

            <div class="queue-content hidden" id="queue-tab-log">
                <div class="activity-log">
                    ${log.length === 0 ? `
                        <div class="queue-empty">
                            <p>No activity yet</p>
                        </div>
                    ` : log.map(entry => `
                        <div class="log-entry">
                            <span class="log-time">${this.escapeHtml(entry.time)}</span>
                            <span class="log-message">${this.escapeHtml(entry.message)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="queue-footer">
                <button class="queue-btn secondary" id="queue-clear-processed">Clear Processed</button>
                <button class="queue-btn secondary" id="queue-dismiss-all">Dismiss All</button>
                <button class="queue-btn primary" id="queue-approve-all">Approve All</button>
            </div>
        `;
    },

    renderQueueItem(item) {
        const priorityClass = `priority-${item.priority || 'medium'}`;
        const alignmentClass = `alignment-${item.manifestation_alignment || 'medium'}`;
        const typeIcon = {
            'insight': '💡',
            'improvement': '⚡',
            'connection': '🔗',
            'code_issue': '🐛'
        }[item.type] || '📝';

        // Alignment indicator with visual cue
        const alignmentIcon = {
            'high': '🎯',
            'medium': '◉',
            'low': '○'
        }[item.manifestation_alignment] || '◉';

        const timeAgo = this.formatTimeAgo(item.timestamp);

        // Escape all user-controlled content to prevent XSS
        const safeTitle = this.escapeHtml(item.title) || 'Untitled';
        const safeDescription = this.escapeHtml(item.description);
        const safePriority = this.escapeHtml(item.priority) || 'medium';
        const safeAlignment = this.escapeHtml(item.manifestation_alignment) || 'medium';
        const safeVisionConnection = this.escapeHtml(item.vision_connection);
        const safeRelatedNodes = item.relatedNodes?.slice(0, 3).map(n => this.escapeHtml(n)).join(', ');
        const safeId = this.escapeHtml(item.id);

        return `
            <div class="queue-item ${priorityClass} ${alignmentClass}" data-id="${safeId}">
                <div class="queue-item-header">
                    <span class="queue-item-type">${typeIcon}</span>
                    <span class="queue-item-title">${safeTitle}</span>
                    <span class="queue-item-alignment" title="Vision Alignment: ${safeAlignment}">${alignmentIcon}</span>
                    <span class="queue-item-priority">${safePriority}</span>
                </div>
                <div class="queue-item-body">
                    <p class="queue-item-description">${safeDescription}</p>
                    ${safeVisionConnection ? `
                        <div class="queue-item-vision">
                            ↳ ${safeVisionConnection}
                        </div>
                    ` : ''}
                    ${item.relatedNodes?.length ? `
                        <div class="queue-item-related">
                            Related: ${safeRelatedNodes}
                        </div>
                    ` : ''}
                </div>
                <div class="queue-item-footer">
                    <span class="queue-item-time">${timeAgo}</span>
                    <div class="queue-item-actions">
                        <button class="queue-action-btn dismiss" data-action="dismiss" data-id="${safeId}">Dismiss</button>
                        <button class="queue-action-btn approve" data-action="approve" data-id="${safeId}">Approve</button>
                    </div>
                </div>
            </div>
        `;
    },

    attachQueueListeners() {
        // Close button
        const closeBtn = document.getElementById('queue-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hideQueuePanel());
        }

        // Tab switching
        document.querySelectorAll('.queue-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.queue-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                const tabName = tab.dataset.tab;
                document.querySelectorAll('.queue-content').forEach(content => {
                    content.classList.add('hidden');
                });
                document.getElementById(`queue-tab-${tabName}`)?.classList.remove('hidden');
            });
        });

        // Item actions (event delegation)
        const itemsContainer = document.querySelector('.queue-items');
        if (itemsContainer) {
            itemsContainer.addEventListener('click', async (e) => {
                const btn = e.target.closest('.queue-action-btn');
                if (!btn) return;

                const action = btn.dataset.action;
                const id = btn.dataset.id;

                btn.disabled = true;

                if (action === 'approve') {
                    await ReflectionDaemon.approveItem(id);
                } else if (action === 'dismiss') {
                    await ReflectionDaemon.dismissItem(id);
                }
            });
        }

        // Footer buttons
        const clearBtn = document.getElementById('queue-clear-processed');
        if (clearBtn) {
            clearBtn.addEventListener('click', async () => {
                await ReflectionDaemon.clearProcessed();
                this.refreshQueueView();
            });
        }

        const dismissAllBtn = document.getElementById('queue-dismiss-all');
        if (dismissAllBtn) {
            dismissAllBtn.addEventListener('click', async () => {
                try {
                    dismissAllBtn.disabled = true;
                    const items = await ReflectionDaemon.getQueue({ status: 'pending' });
                    // Use Promise.all for parallel processing
                    await Promise.all(items.map(item => ReflectionDaemon.dismissItem(item.id)));
                    this.refreshQueueView();
                } catch (error) {
                    console.error('Failed to dismiss all items:', error);
                } finally {
                    dismissAllBtn.disabled = false;
                }
            });
        }

        const approveAllBtn = document.getElementById('queue-approve-all');
        if (approveAllBtn) {
            approveAllBtn.addEventListener('click', async () => {
                try {
                    approveAllBtn.disabled = true;
                    const items = await ReflectionDaemon.getQueue({ status: 'pending' });
                    // Use Promise.all for parallel processing
                    await Promise.all(items.map(item => ReflectionDaemon.approveItem(item.id)));
                    this.refreshQueueView();
                } catch (error) {
                    console.error('Failed to approve all items:', error);
                } finally {
                    approveAllBtn.disabled = false;
                }
            });
        }

        // Close on escape - clean up previous handler first
        if (this._boundListeners.escapeHandler) {
            document.removeEventListener('keydown', this._boundListeners.escapeHandler);
        }
        this._boundListeners.escapeHandler = (e) => {
            if (e.key === 'Escape' && this.panelVisible) {
                this.hideQueuePanel();
            }
        };
        document.addEventListener('keydown', this._boundListeners.escapeHandler);

        // Close on outside click
        const panel = document.getElementById('reflection-queue-panel');
        if (panel) {
            panel.addEventListener('click', (e) => {
                if (e.target === panel) {
                    this.hideQueuePanel();
                }
            });
        }
    },

    async refreshQueueView() {
        if (!this.panelVisible) return;

        const content = document.getElementById('queue-tab-pending');
        if (content) {
            const items = await ReflectionDaemon.getQueue({ status: 'pending' });

            // Update tab count
            const tab = document.querySelector('.queue-tab[data-tab="pending"]');
            if (tab) {
                tab.textContent = `Pending (${items.length})`;
            }

            // Update content
            if (items.length === 0) {
                content.innerHTML = `
                    <div class="queue-empty">
                        <div class="queue-empty-icon">🔮</div>
                        <p>No pending reflections</p>
                        <p class="queue-empty-hint">Reflections will appear here when the AI analyzes your map</p>
                    </div>
                `;
            } else {
                content.innerHTML = `
                    <div class="queue-items">
                        ${items.map(item => this.renderQueueItem(item)).join('')}
                    </div>
                `;
            }

            // Re-attach item listeners
            this.attachQueueListeners();
        }

        this.updateBadge();
    },

    // ═══════════════════════════════════════════════════════════════════
    // UTILITIES
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Escape HTML to prevent XSS attacks
     */
    escapeHtml(str) {
        if (!str || typeof str !== 'string') return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },

    formatTimeAgo(timestamp) {
        const seconds = Math.floor((Date.now() - timestamp) / 1000);

        if (seconds < 60) return 'just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }
};

// ═══════════════════════════════════════════════════════════════════
// GITHUB CONFIG UI FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/**
 * Toggle GitHub config panel visibility
 */
function toggleGithubConfig() {
    const panel = document.getElementById('github-config-panel');
    const chevron = document.getElementById('github-config-chevron');

    if (panel && chevron) {
        const isVisible = panel.style.display !== 'none';
        panel.style.display = isVisible ? 'none' : 'block';
        chevron.style.transform = isVisible ? '' : 'rotate(180deg)';

        // Load existing config if opening
        if (!isVisible) {
            loadGithubConfig();
        }
    }
}

/**
 * Load existing GitHub config into form
 */
function loadGithubConfig() {
    if (typeof ReflectionDaemon === 'undefined') return;

    const status = ReflectionDaemon.getStatus();
    const github = status.github || {};

    const repoInput = document.getElementById('github-repo-input');
    const branchInput = document.getElementById('github-branch-input');
    const tokenInput = document.getElementById('github-token-input');

    if (repoInput && github.owner && github.repo) {
        repoInput.value = `${github.owner}/${github.repo}`;
    }

    if (branchInput && ReflectionDaemon.config?.github?.baseBranch) {
        branchInput.value = ReflectionDaemon.config.github.baseBranch;
    }

    // Token is stored securely, show placeholder if configured
    if (tokenInput && github.configured) {
        tokenInput.placeholder = '••••••••••••••••';
    }

    updateGithubStatusIndicator(github.configured);
}

/**
 * Save GitHub config
 */
function saveGithubConfig() {
    if (typeof ReflectionDaemon === 'undefined') {
        showGithubStatus('ReflectionDaemon not loaded', 'error');
        return;
    }

    const repoInput = document.getElementById('github-repo-input');
    const tokenInput = document.getElementById('github-token-input');
    const branchInput = document.getElementById('github-branch-input');

    const repoValue = repoInput?.value?.trim() || '';
    const tokenValue = tokenInput?.value?.trim() || '';
    const branchValue = branchInput?.value?.trim() || 'main';

    // Parse owner/repo
    const repoParts = repoValue.split('/');
    if (repoParts.length !== 2) {
        showGithubStatus('Invalid format. Use: owner/repo', 'error');
        return;
    }

    const [owner, repo] = repoParts;

    // Get existing token if not provided
    const existingToken = ReflectionDaemon.githubToken;
    const token = tokenValue || existingToken;

    if (!token) {
        showGithubStatus('Token is required', 'error');
        return;
    }

    try {
        const result = ReflectionDaemon.configureGithub({
            owner: owner.trim(),
            repo: repo.trim(),
            token: token,
            baseBranch: branchValue
        });

        if (result.enabled) {
            showGithubStatus('GitHub connected successfully!', 'success');
            updateGithubStatusIndicator(true);
            // Clear token field for security
            if (tokenInput) tokenInput.value = '';
            if (tokenInput) tokenInput.placeholder = '••••••••••••••••';
        } else {
            showGithubStatus('Configuration saved but not enabled', 'warning');
            updateGithubStatusIndicator(false);
        }
    } catch (error) {
        showGithubStatus(`Error: ${error.message}`, 'error');
        updateGithubStatusIndicator(false);
    }
}

/**
 * Update GitHub status indicator color
 */
function updateGithubStatusIndicator(isConfigured) {
    const indicator = document.getElementById('github-status-indicator');
    if (indicator) {
        indicator.style.background = isConfigured ? '#10b981' : 'var(--text-muted)';
        indicator.title = isConfigured ? 'Connected' : 'Not configured';
    }
}

/**
 * Show status message in GitHub config panel
 */
function showGithubStatus(message, type = 'info') {
    const statusEl = document.getElementById('github-config-status');
    if (!statusEl) return;

    const colors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        info: 'var(--text-muted)'
    };

    statusEl.textContent = message;
    statusEl.style.color = colors[type] || colors.info;

    // Auto-clear after 5 seconds
    setTimeout(() => {
        if (statusEl.textContent === message) {
            statusEl.textContent = '';
        }
    }, 5000);
}

/**
 * Initialize GitHub config on page load
 */
document.addEventListener('DOMContentLoaded', () => {
    // Delay to ensure ReflectionDaemon is loaded
    setTimeout(() => {
        if (typeof ReflectionDaemon !== 'undefined') {
            const status = ReflectionDaemon.getStatus();
            updateGithubStatusIndicator(status.github?.configured || false);
        }
    }, 1000);
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ReflectionUI;
}

console.log('🔮 ReflectionUI loaded. Call ReflectionUI.init() to initialize.');
