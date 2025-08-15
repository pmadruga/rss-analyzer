/**
 * UI Tests for JSON Rendering Functionality
 * Tests the conversion of JSON-formatted article analysis to HTML
 */

const fs = require('fs');
const path = require('path');

describe('JSON rendering functionality', () => {
  let processAnalysisContent;
  let convertJsonAnalysisToHtml;
  let formatJsonKeyAsTitle;
  let cleanJsonText;
  let extractTextFromAnalysis;

  beforeAll(() => {
    // Load the HTML file and extract JavaScript functions
    const htmlPath = path.join(__dirname, '../../docs/index.html');
    
    if (!fs.existsSync(htmlPath)) {
      throw new Error(`HTML file not found at ${htmlPath}`);
    }
    
    const htmlContent = fs.readFileSync(htmlPath, 'utf8');

    // Extract and eval the JavaScript functions from the HTML
    const scriptMatch = htmlContent.match(/<script>(.*?)<\/script>/s);
    if (!scriptMatch) {
      throw new Error('No script tag found in HTML file');
    }
    
    const scriptContent = scriptMatch[1];
    
    // Create a more comprehensive sandbox environment for the functions
    const sandbox = {
      console,
      document: global.document,
      window: global.window,
      // Add any other globals the functions might need
      setTimeout: global.setTimeout,
      setInterval: global.setInterval,
      clearTimeout: global.clearTimeout,
      clearInterval: global.clearInterval,
    };
    
    try {
      // Execute the script content in our test environment
      const vm = require('vm');
      const context = vm.createContext(sandbox);
      vm.runInContext(scriptContent, context);
      
      // Extract the functions we need to test
      processAnalysisContent = context.processAnalysisContent;
      convertJsonAnalysisToHtml = context.convertJsonAnalysisToHtml;
      formatJsonKeyAsTitle = context.formatJsonKeyAsTitle;
      cleanJsonText = context.cleanJsonText;
      extractTextFromAnalysis = context.extractTextFromAnalysis;
      
      // Verify all functions were extracted
      if (!processAnalysisContent) throw new Error('processAnalysisContent function not found');
      if (!convertJsonAnalysisToHtml) throw new Error('convertJsonAnalysisToHtml function not found');
      if (!formatJsonKeyAsTitle) throw new Error('formatJsonKeyAsTitle function not found');
      if (!cleanJsonText) throw new Error('cleanJsonText function not found');
      if (!extractTextFromAnalysis) throw new Error('extractTextFromAnalysis function not found');
      
    } catch (error) {
      console.error('Error loading functions:', error);
      throw new Error(`Failed to load JavaScript functions: ${error.message}`);
    }
  });

  describe('formatJsonKeyAsTitle', () => {
    test('converts snake_case to Title Case', () => {
      expect(formatJsonKeyAsTitle('core_concept')).toBe('Core Concept');
      expect(formatJsonKeyAsTitle('feynman_technique_breakdown')).toBe('Feynman Technique Breakdown');
    });

    test('converts camelCase to Title Case', () => {
      expect(formatJsonKeyAsTitle('coreConceptExplanation')).toBe('Core Concept Explanation');
      expect(formatJsonKeyAsTitle('stepBasedAnalysis')).toBe('Step Based Analysis');
    });

    test('handles mixed formats', () => {
      expect(formatJsonKeyAsTitle('step_1_simple_explanation')).toBe('Step 1 Simple Explanation');
      expect(formatJsonKeyAsTitle('why_this_matters')).toBe('Why This Matters');
    });

    test('handles edge cases', () => {
      expect(formatJsonKeyAsTitle('')).toBe('');
      expect(formatJsonKeyAsTitle('single')).toBe('Single');
      expect(formatJsonKeyAsTitle('ALLCAPS')).toBe('A L L C A P S');
    });
  });

  describe('cleanJsonText', () => {
    test('converts markdown formatting to HTML', () => {
      const input = '**bold text** and *italic text* and `code text`';
      const result = cleanJsonText(input);
      expect(result).toContain('<strong>bold text</strong>');
      expect(result).toContain('<em>italic text</em>');
      expect(result).toContain('<code>code text</code>');
    });

    test('handles newlines correctly', () => {
      const input = 'Line 1\n\nLine 2\nLine 3';
      const result = cleanJsonText(input);
      expect(result).toContain('Line 1');
      expect(result).toContain('Line 2');
      expect(result).toContain('Line 3');
    });

    test('returns non-string inputs unchanged', () => {
      expect(cleanJsonText(123)).toBe(123);
      expect(cleanJsonText(null)).toBe(null);
      expect(cleanJsonText(undefined)).toBe(undefined);
    });

    test('handles empty string', () => {
      expect(cleanJsonText('')).toBe('');
    });
  });

  describe('processAnalysisContent', () => {
    test('detects and processes JSON content', () => {
      const jsonInput = `\`\`\`json
{
  "extracted_title": "Test Article",
  "analysis": {
    "core_concept": "This is a test concept"
  }
}
\`\`\``;

      const result = processAnalysisContent(jsonInput);
      expect(result).toContain('Test Article');
      expect(result).toContain('Core Concept');
      expect(result).toContain('This is a test concept');
    });

    test('processes regular markdown content', () => {
      const markdownInput = '**Bold text** with regular content.';
      const result = processAnalysisContent(markdownInput);
      expect(result).toContain('<strong>Bold text</strong>');
      expect(result).toContain('<p>');
    });

    test('handles malformed JSON gracefully', () => {
      const malformedJson = `\`\`\`json
{ "invalid": json }
\`\`\``;

      // Should not throw error and should fall back to markdown processing
      const result = processAnalysisContent(malformedJson);
      expect(result).toBeTruthy();
      expect(typeof result).toBe('string');
    });

    test('handles empty content', () => {
      expect(processAnalysisContent('')).toBeTruthy();
      expect(processAnalysisContent('   ')).toBeTruthy();
    });
  });

  describe('convertJsonAnalysisToHtml', () => {
    test('handles simple JSON structure', () => {
      const jsonData = {
        extracted_title: "Simple Test",
        analysis: {
          basic_concept: "This is a basic concept"
        }
      };

      const result = convertJsonAnalysisToHtml(jsonData);
      expect(result).toContain('Simple Test');
      expect(result).toContain('Basic Concept');
      expect(result).toContain('This is a basic concept');
    });

    test('handles complex nested JSON structure', () => {
      const jsonData = {
        extracted_title: "Complex Analysis Test",
        analysis: {
          feynman_technique_breakdown: {
            core_concept: {
              simple_explanation: "This is a simple explanation",
              analogy: "Like a simple analogy"
            }
          }
        }
      };

      const result = convertJsonAnalysisToHtml(jsonData);
      expect(result).toContain('Complex Analysis Test');
      expect(result).toContain('Core Concept');
      expect(result).toContain('This is a simple explanation');
      expect(result).toContain('Like a simple analogy');
    });

    test('handles arrays correctly', () => {
      const jsonData = {
        analysis: {
          simple_array: ["Item 1", "Item 2", "Item 3"]
        }
      };

      const result = convertJsonAnalysisToHtml(jsonData);
      expect(result).toContain('Simple Array');
      expect(result).toContain('Item 1');
      expect(result).toContain('Item 2');
      expect(result).toContain('Item 3');
    });

    test('returns fallback message for empty or invalid data', () => {
      expect(convertJsonAnalysisToHtml({})).toContain('Analysis content could not be displayed properly');
      expect(convertJsonAnalysisToHtml(null)).toContain('Analysis content could not be displayed properly');
      expect(convertJsonAnalysisToHtml(undefined)).toContain('Analysis content could not be displayed properly');
    });
  });

  describe('extractTextFromAnalysis', () => {
    test('extracts text from JSON content', () => {
      const jsonInput = `\`\`\`json
{
  "extracted_title": "Test Title",
  "analysis": {
    "concept": "Test concept text"
  }
}
\`\`\``;

      const result = extractTextFromAnalysis(jsonInput);
      expect(result).toContain('Test Title');
      expect(result).toContain('Test concept text');
    });

    test('extracts text from markdown content', () => {
      const markdownInput = '**Bold text** with regular content.';
      const result = extractTextFromAnalysis(markdownInput);
      expect(result).toContain('Bold text');
      expect(result).toContain('with regular content');
    });

    test('handles malformed JSON gracefully', () => {
      const malformedJson = `\`\`\`json
{ invalid json }
\`\`\``;

      const result = extractTextFromAnalysis(malformedJson);
      expect(typeof result).toBe('string');
      expect(result.length).toBeGreaterThan(0);
    });
  });

  describe('Integration tests', () => {
    test('full JSON processing pipeline with real-world structure', () => {
      const realWorldInput = `\`\`\`json
{
  "extracted_title": "Context Engineering for Agents: Write, Select, Compress, and Isolate",
  "analysis": {
    "feynman_technique_breakdown": {
      "core_concept": {
        "simple_explanation": "Context engineering is like managing a computer's RAM for an AI agent.",
        "analogy": "Imagine you're a chef in a tiny kitchen (the context window)."
      },
      "four_strategies_deep_dive": {
        "1_write_context": {
          "what": "Storing context outside the agent's active memory",
          "why": "Prevents the context window from filling up"
        }
      }
    },
    "practical_examples": [
      "Claude Code saves plans to memory",
      "ChatGPT uses embeddings for retrieval"
    ]
  }
}
\`\`\``;

      const result = processAnalysisContent(realWorldInput);
      
      // Check main title
      expect(result).toContain('Context Engineering for Agents');
      
      // Check core concept section
      expect(result).toContain('Core Concept');
      expect(result).toContain("Context engineering is like managing a computer's RAM");
      
      // Check nested structures
      expect(result).toContain('Write Context');
      expect(result).toContain('Storing context outside');
      
      // Check arrays
      expect(result).toContain('Practical Examples');
      expect(result).toContain('Claude Code saves plans');
      expect(result).toContain('ChatGPT uses embeddings');
    });
  });
});