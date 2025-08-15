/**
 * UI Tests for JSON Rendering Functionality
 * Tests the conversion of JSON-formatted article analysis to HTML
 */

describe('JSON rendering functionality', () => {
  let processAnalysisContent;
  let convertJsonAnalysisToHtml;
  let formatJsonKeyAsTitle;
  let cleanJsonText;

  beforeAll(() => {
    // Load the HTML file and extract JavaScript functions
    const fs = require('fs');
    const path = require('path');
    const htmlContent = fs.readFileSync(
      path.join(__dirname, '../../docs/index.html'),
      'utf8'
    );

    // Extract and eval the JavaScript functions from the HTML
    const scriptMatch = htmlContent.match(/<script>(.*?)<\/script>/s);
    if (scriptMatch) {
      const scriptContent = scriptMatch[1];
      
      // Create a sandbox environment for the functions
      const sandbox = {
        console,
        document: global.document,
        window: global.window
      };
      
      // Execute the script content in our test environment
      const vm = require('vm');
      const context = vm.createContext(sandbox);
      vm.runInContext(scriptContent, context);
      
      // Extract the functions we need to test
      processAnalysisContent = context.processAnalysisContent;
      convertJsonAnalysisToHtml = context.convertJsonAnalysisToHtml;
      formatJsonKeyAsTitle = context.formatJsonKeyAsTitle;
      cleanJsonText = context.cleanJsonText;
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
  });

  describe('cleanJsonText', () => {
    test('converts markdown formatting to HTML', () => {
      const input = '**bold text** and *italic text* and `code text`';
      const expected = '<strong>bold text</strong> and <em>italic text</em> and <code>code text</code>';
      expect(cleanJsonText(input)).toBe(expected);
    });

    test('handles newlines correctly', () => {
      const input = 'Line 1\n\nLine 2\nLine 3';
      const expected = 'Line 1</p><p>Line 2 Line 3';
      expect(cleanJsonText(input)).toBe(expected);
    });

    test('returns non-string inputs unchanged', () => {
      expect(cleanJsonText(123)).toBe(123);
      expect(cleanJsonText(null)).toBe(null);
      expect(cleanJsonText(undefined)).toBe(undefined);
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
      expect(result).toContain('<h3>Test Article</h3>');
      expect(result).toContain('<h4>Core Concept</h4>');
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

      const result = processAnalysisContent(malformedJson);
      // Should fall back to markdown processing
      expect(result).toBeTruthy();
    });
  });

  describe('convertJsonAnalysisToHtml', () => {
    test('handles complex nested JSON structure', () => {
      const jsonData = {
        extracted_title: "Complex Analysis Test",
        analysis: {
          feynman_technique_breakdown: {
            core_concept: {
              simple_explanation: "This is a simple explanation",
              analogy: "Like a simple analogy"
            },
            four_strategies_deep_dive: {
              strategy_1: {
                what: "What it does",
                why: "Why it matters",
                how: ["Step 1", "Step 2", "Step 3"]
              }
            }
          }
        }
      };

      const result = convertJsonAnalysisToHtml(jsonData);
      
      // Check for title
      expect(result).toContain('<h3>Complex Analysis Test</h3>');
      
      // Check for core concept processing
      expect(result).toContain('<h4>Core Concept</h4>');
      expect(result).toContain('This is a simple explanation');
      
      // Check for analogy
      expect(result).toContain('<h5>Analogy</h5>');
      expect(result).toContain('Like a simple analogy');
      
      // Check for nested structures
      expect(result).toContain('What');
      expect(result).toContain('Why');
      expect(result).toContain('How');
    });

    test('handles step-based analysis structure', () => {
      const jsonData = {
        analysis: {
          step_1_simple_explanation: {
            core_concept: "Step 1 concept",
            details: "Additional details"
          },
          step_2_breakdown: {
            components: ["Component A", "Component B"]
          }
        }
      };

      const result = convertJsonAnalysisToHtml(jsonData);
      
      expect(result).toContain('<h4>Step 1 Simple Explanation</h4>');
      expect(result).toContain('<h4>Step 2 Breakdown</h4>');
      expect(result).toContain('Step 1 concept');
      expect(result).toContain('Component A');
      expect(result).toContain('Component B');
    });

    test('handles arrays with mixed content types', () => {
      const jsonData = {
        analysis: {
          mixed_array: [
            "Simple string item",
            {
              name: "Complex Object",
              description: "Object description"
            }
          ]
        }
      };

      const result = convertJsonAnalysisToHtml(jsonData);
      
      expect(result).toContain('<h3>Mixed Array</h3>');
      expect(result).toContain('Simple string item');
      expect(result).toContain('Complex Object');
      expect(result).toContain('Object description');
    });

    test('returns fallback message for empty or invalid data', () => {
      expect(convertJsonAnalysisToHtml({})).toContain('Analysis content could not be displayed properly');
      expect(convertJsonAnalysisToHtml(null)).toContain('Analysis content could not be displayed properly');
    });
  });

  describe('Integration tests', () => {
    test('full JSON processing pipeline', () => {
      const complexJsonInput = `\`\`\`json
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
          "why": "Prevents the context window from filling up",
          "how": {
            "scratchpads": "Temporary storage for task-specific notes",
            "memories": "Long-term storage for reusable knowledge"
          }
        }
      }
    },
    "why_this_matters": {
      "problems_solved": [
        "Context poisoning",
        "Context distraction",
        "Context confusion"
      ]
    }
  }
}
\`\`\``;

      const result = processAnalysisContent(complexJsonInput);
      
      // Check main title
      expect(result).toContain('<h3>Context Engineering for Agents: Write, Select, Compress, and Isolate</h3>');
      
      // Check core concept section
      expect(result).toContain('<h4>Core Concept</h4>');
      expect(result).toContain("Context engineering is like managing a computer's RAM");
      
      // Check analogy
      expect(result).toContain('<h5>Analogy</h5>');
      expect(result).toContain("chef in a tiny kitchen");
      
      // Check nested structures
      expect(result).toContain('Write Context');
      expect(result).toContain('Storing context outside');
      expect(result).toContain('Scratchpads');
      expect(result).toContain('Memories');
      
      // Check arrays
      expect(result).toContain('Why This Matters');
      expect(result).toContain('Context poisoning');
      expect(result).toContain('Context distraction');
    });

    test('handles real-world article structure from deployment', () => {
      // This tests the actual structure we see in the deployed data
      const realWorldInput = `\`\`\`json
{
    "extracted_title": "GlórIA: A Generative and Open Large Language Model for Portuguese",
    "analysis": {
        "step_1_simple_explanation": {
            "core_idea": "This paper introduces GlórIA, the first open-source, generative large language model (LLM) specifically trained for Portuguese from scratch.",
            "key_components": [
                {
                    "component": "Architecture",
                    "explanation": "GlórIA uses a decoder-only transformer (like GPT), but with modifications tailored to Portuguese"
                },
                {
                    "component": "Training Data",
                    "explanation": "Curated 1.1B-token corpus from diverse Portuguese sources"
                }
            ]
        }
    }
}
\`\`\``;

      const result = processAnalysisContent(realWorldInput);
      
      expect(result).toContain('<h3>GlórIA: A Generative and Open Large Language Model for Portuguese</h3>');
      expect(result).toContain('<h4>Step 1 Simple Explanation</h4>');
      expect(result).toContain('open-source, generative large language model');
      expect(result).toContain('Architecture');
      expect(result).toContain('Training Data');
      expect(result).toContain('decoder-only transformer');
      expect(result).toContain('1.1B-token corpus');
    });
  });
});