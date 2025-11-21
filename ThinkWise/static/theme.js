// Theme management
class ThemeManager {
  constructor() {
    this.theme = localStorage.getItem('thinkwise-theme') || 'light';
    this.init();
  }

  init() {
    this.applyTheme(this.theme);
    this.bindEvents();
  }

  applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('thinkwise-theme', theme);
    this.updateToggleButton(theme);
  }

  toggleTheme() {
    this.theme = this.theme === 'light' ? 'dark' : 'light';
    this.applyTheme(this.theme);
  }

  updateToggleButton(theme) {
    const button = document.querySelector('.theme-toggle');
    if (button) {
      const icon = theme === 'dark' ? '☀️' : '🌙';
      const text = theme === 'dark' ? 'Light Mode' : 'Dark Mode';
      button.innerHTML = `${icon} ${text}`;
    }
  }

  bindEvents() {
    const toggleButton = document.querySelector('.theme-toggle');
    if (toggleButton) {
      toggleButton.addEventListener('click', () => this.toggleTheme());
    }
  }
}

// Initialize theme manager when DOM loads
document.addEventListener('DOMContentLoaded', () => {
  new ThemeManager();
});