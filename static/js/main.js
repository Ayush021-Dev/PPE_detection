// Main JavaScript file for Safety Detection System

// Global variables
let statusCheckInterval;
let alertsCheckInterval;
let isConnected = true;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Check if we're on the main page
    if (document.getElementById('video-feed')) {
        setupVideoFeed();
        startPeriodicChecks();
    }
    
    // Setup common event listeners
    setupEventListeners();
}

function setupVideoFeed() {
    const videoFeed = document.getElementById('video-feed');
    const statusBadge = document.getElementById('status-badge');
    
    // Handle video feed errors
    videoFeed.addEventListener('error', function() {
        console.error('Video feed error');
        updateConnectionStatus(false);
    });
    
    // Handle video feed load
    videoFeed.addEventListener('load', function() {
        updateConnectionStatus(true);
    });
    
    // Check if video is loading
    videoFeed.addEventListener('loadstart', function() {
        console.log('Video feed loading...');
    });
}

function setupEventListeners() {
    // Handle window focus/blur for performance
    window.addEventListener('focus', function() {
        if (!statusCheckInterval) {
            startPeriodicChecks();
        }
    });
    
    window.addEventListener('blur', function() {
        // Optional: pause checks when window is not focused
        // clearPeriodicChecks();
    });
}

function startPeriodicChecks() {
    // Clear existing intervals
    clearPeriodicChecks();
    
    // Start new intervals
    statusCheckInterval = setInterval(checkSystemStatus, 10000); // Every 10 seconds
    alertsCheckInterval = setInterval(loadRecentAlerts, 5000);   // Every 5 seconds
    
    // Initial checks
    checkSystemStatus();
    loadRecentAlerts();
}

function clearPeriodicChecks() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
    if (alertsCheckInterval) {
        clearInterval(alertsCheckInterval);
        alertsCheckInterval = null;
    }
}

function checkSystemStatus() {
    fetch('/api/status')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            updateSystemStatus(data);
            updateConnectionStatus(true);
        })
        .catch(error => {
            console.error('Status check failed:', error);
            updateConnectionStatus(false);
        });
}

function updateSystemStatus(data) {
    // Update model status
    const modelStatus = document.getElementById('model-status');
    if (modelStatus) {
        modelStatus.textContent = data.model_loaded ? 'Loaded' : 'Error';
        modelStatus.className = data.model_loaded ? 'badge bg-success' : 'badge bg-danger';
    }
    
    // Update camera status
    const cameraStatus = document.getElementById('camera-status');
    if (cameraStatus) {
        cameraStatus.textContent = data.camera_active ? 'Active' : 'Inactive';
        cameraStatus.className = data.camera_active ? 'badge bg-success' : 'badge bg-danger';
    }
    
    // Update last update time
    const lastUpdate = document.getElementById('last-update');
    if (lastUpdate) {
        lastUpdate.textContent = new Date().toLocaleTimeString();
    }
}

function updateConnectionStatus(connected) {
    isConnected = connected;
    const statusBadge = document.getElementById('status-badge');
    const statusLight = document.getElementById('status-light');
    
    if (statusBadge) {
        if (connected) {
            statusBadge.innerHTML = '<i class="fas fa-circle me-1"></i>Live';
            statusBadge.className = 'badge bg-success';
        } else {
            statusBadge.innerHTML = '<i class="fas fa-exclamation-circle me-1"></i>Disconnected';
            statusBadge.className = 'badge bg-danger';
        }
    }
    
    if (statusLight) {
        statusLight.className = connected ? 'status-light safe' : 'status-light danger';
    }
}

function loadRecentAlerts() {
    fetch('/api/logs')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            updateRecentAlerts(data);
            updateDetectionCounts(data);
        })
        .catch(error => {
            console.error('Failed to load alerts:', error);
        });
}

function updateRecentAlerts(logs) {
    const container = document.getElementById('recent-alerts');
    if (!container) return;
    
    if (logs.length === 0) {
        container.innerHTML = '<p class="text-muted">No recent alerts</p>';
    } else {
        const recentLogs = logs.slice(0, 5);
        container.innerHTML = recentLogs.map(log => {
            const alertClass = log.warning_level >= 2 ? 'danger' : (log.warning_level === 1 ? 'warning' : 'success');
            const icon = log.warning_level >= 2 ? 'exclamation-triangle' : (log.warning_level === 1 ? 'exclamation-circle' : 'check-circle');
            
            return `
                <div class="alert alert-${alertClass} py-2 mb-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <i class="fas fa-${icon} me-2"></i>
                            <span>${log.categories || 'Safety check'}</span>
                        </div>
                        <small>${formatTimestamp(log.timestamp)}</small>
                    </div>
                </div>
            `;
        }).join('');
    }
}

function updateDetectionCounts(logs) {
    // This is a simplified version - you might want to track real-time detections
    const helmetCount = document.getElementById('helmet-count');
    const boilerCount = document.getElementById('boiler-count');
    
    if (helmetCount && boilerCount) {
        // For demo purposes - in real implementation, this should come from the detection system
        helmetCount.textContent = Math.floor(Math.random() * 3);
        boilerCount.textContent = Math.floor(Math.random() * 3);
    }
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) { // Less than 1 minute
        return 'Just now';
    } else if (diff < 3600000) { // Less than 1 hour
        return Math.floor(diff / 60000) + 'm ago';
    } else if (diff < 86400000) { // Less than 1 day
        return Math.floor(diff / 3600000) + 'h ago';
    } else {
        return date.toLocaleDateString();
    }
}

// Utility functions
function showNotification(message, type = 'info') {
    // Create toast notification (if needed)
    console.log(`${type.toUpperCase()}: ${message}`);
}

function debounce(func, wait) {
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

// Cleanup when page is unloaded
window.addEventListener('beforeunload', function() {
    clearPeriodicChecks();
});