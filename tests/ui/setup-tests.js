// Setup file for Jest tests
const { getByRole, getByText } = require('@testing-library/dom');
require('@testing-library/jest-dom');

// Mock fetch for testing
global.fetch = jest.fn();

// Mock console.warn to avoid noise in tests
global.console.warn = jest.fn();

// Mock window object for DOM tests
Object.defineProperty(window, 'location', {
  value: {
    href: 'http://localhost/',
    protocol: 'http:',
    host: 'localhost',
    pathname: '/',
  },
  writable: true,
});

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock;

// Setup DOM environment
beforeEach(() => {
  document.body.innerHTML = '';
  fetch.mockClear();
  console.warn.mockClear();
  
  // Clear localStorage mock
  localStorageMock.getItem.mockClear();
  localStorageMock.setItem.mockClear();
  localStorageMock.removeItem.mockClear();
  localStorageMock.clear.mockClear();
});