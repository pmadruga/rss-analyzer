// Setup file for Jest tests
import '@testing-library/jest-dom';

// Mock fetch for testing
global.fetch = jest.fn();

// Mock console.warn to avoid noise in tests
global.console.warn = jest.fn();

// Setup DOM environment
beforeEach(() => {
  document.body.innerHTML = '';
  fetch.mockClear();
  console.warn.mockClear();
});