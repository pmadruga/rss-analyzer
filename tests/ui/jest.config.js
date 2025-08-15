module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/setup-tests.js'],
  testMatch: ['<rootDir>/**/*.test.js'],
  moduleFileExtensions: ['js', 'json'],
  transform: {},
  testTimeout: 30000,
  collectCoverageFrom: [
    '../../docs/**/*.{js,html}'
  ],
  coverageReporters: ['text', 'json-summary', 'html'],
  verbose: true,
  // Ignore coverage for HTML files since they're not JavaScript
  coveragePathIgnorePatterns: [
    '../../docs/.*\\.html$'
  ],
  // Add some helpful options for CI
  maxWorkers: 1,
  detectOpenHandles: true,
  forceExit: true
};