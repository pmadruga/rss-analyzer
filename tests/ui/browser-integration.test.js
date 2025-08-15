/**
 * Browser Integration Tests for JSON Rendering
 * Tests the actual website functionality in a real browser environment
 */

const puppeteer = require('puppeteer');
const path = require('path');

describe('Browser Integration - JSON Rendering', () => {
  let browser;
  let page;

  beforeAll(async () => {
    browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
  });

  afterAll(async () => {
    if (browser) {
      await browser.close();
    }
  });

  beforeEach(async () => {
    page = await browser.newPage();
    
    // Mock the data.json API call with test data
    await page.setRequestInterception(true);
    page.on('request', (request) => {
      if (request.url().includes('data.json')) {
        request.respond({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            generated_at: new Date().toISOString(),
            total_articles: 2,
            articles: [
              {
                id: 1,
                title: "JSON Test Article",
                url: "https://example.com/test",
                processed_date: "2025-08-15 10:00:00",
                status: "completed",
                analysis: `\`\`\`json
{
  "extracted_title": "Context Engineering for Agents",
  "analysis": {
    "feynman_technique_breakdown": {
      "core_concept": {
        "simple_explanation": "Context engineering is like managing a computer's RAM for an AI agent.",
        "analogy": "Imagine you're a chef in a tiny kitchen."
      },
      "four_strategies": {
        "write": "Store context externally",
        "select": "Choose relevant information",
        "compress": "Reduce context size",
        "isolate": "Separate different contexts"
      }
    },
    "practical_examples": [
      "Claude Code saves plans to memory",
      "ChatGPT uses embeddings for retrieval",
      "Multi-agent systems use isolation"
    ]
  }
}
\`\`\``,
                ai_provider: "anthropic",
                linked_articles: []
              },
              {
                id: 2,
                title: "Markdown Test Article",
                url: "https://example.com/markdown",
                processed_date: "2025-08-15 09:00:00",
                status: "completed",
                analysis: "### Traditional Markdown Analysis\n\nThis is a **traditional markdown** analysis with *italic* text and regular paragraphs.\n\n- List item 1\n- List item 2\n- List item 3",
                ai_provider: "mistral",
                linked_articles: []
              }
            ]
          })
        });
      } else {
        request.continue();
      }
    });

    // Load the HTML file
    const htmlPath = path.join(__dirname, '../../docs/index.html');
    await page.goto(`file://${htmlPath}`);
    
    // Wait for the page to load and process articles
    await page.waitForSelector('.article-card', { timeout: 5000 });
  });

  afterEach(async () => {
    if (page) {
      await page.close();
    }
  });

  test('loads and displays articles correctly', async () => {
    // Check that articles are displayed
    const articles = await page.$$('.article-card');
    expect(articles).toHaveLength(2);

    // Check article titles
    const titles = await page.$$eval('.article-title a', els => els.map(el => el.textContent));
    expect(titles).toContain('JSON Test Article');
    expect(titles).toContain('Markdown Test Article');
  });

  test('JSON article displays structured content with proper headers', async () => {
    // Find the JSON test article
    const jsonArticle = await page.evaluateHandle(() => {
      const cards = Array.from(document.querySelectorAll('.article-card'));
      return cards.find(card => 
        card.querySelector('.article-title a').textContent === 'JSON Test Article'
      );
    });

    // Check that JSON content is processed and displayed as HTML
    const analysisContent = await jsonArticle.evaluate(el => 
      el.querySelector('.article-analysis').innerHTML
    );

    // Should contain structured headers, not raw JSON
    expect(analysisContent).toContain('<h3>Context Engineering for Agents</h3>');
    expect(analysisContent).toContain('<h4>Core Concept</h4>');
    expect(analysisContent).toContain('<h5>Simple Explanation</h5>');
    expect(analysisContent).toContain('Context engineering is like managing');
    
    // Should not contain raw JSON
    expect(analysisContent).not.toContain('```json');
    expect(analysisContent).not.toContain('"extracted_title"');
    expect(analysisContent).not.toContain('"feynman_technique_breakdown"');
  });

  test('traditional markdown article displays correctly', async () => {
    const markdownArticle = await page.evaluateHandle(() => {
      const cards = Array.from(document.querySelectorAll('.article-card'));
      return cards.find(card => 
        card.querySelector('.article-title a').textContent === 'Markdown Test Article'
      );
    });

    const analysisContent = await markdownArticle.evaluate(el => 
      el.querySelector('.article-analysis').innerHTML
    );

    // Should contain processed markdown
    expect(analysisContent).toContain('<strong>traditional markdown</strong>');
    expect(analysisContent).toContain('<em>italic</em>');
    expect(analysisContent).toContain('<p>');
  });

  test('fullscreen modal works with JSON content', async () => {
    // Click the fullscreen button for the JSON article
    await page.click('.article-card .fullscreen-toggle');

    // Wait for modal to appear
    await page.waitForSelector('#fullscreen-modal[style*="block"]', { timeout: 2000 });

    // Check modal title
    const modalTitle = await page.$eval('#fullscreen-title a', el => el.textContent);
    expect(modalTitle).toBe('JSON Test Article');

    // Check that modal content is properly formatted
    const modalContent = await page.$eval('#fullscreen-analysis', el => el.innerHTML);
    expect(modalContent).toContain('<h3>Context Engineering for Agents</h3>');
    expect(modalContent).toContain('<h4>Core Concept</h4>');
    expect(modalContent).not.toContain('```json');

    // Close modal
    await page.click('#close-fullscreen');
    
    // Wait for modal to disappear
    await page.waitForFunction(() => {
      const modal = document.querySelector('#fullscreen-modal');
      return modal.style.display === 'none' || modal.style.display === '';
    });
  });

  test('expand/collapse functionality works with JSON content', async () => {
    // Find an article with expand functionality
    const expandButton = await page.$('.expand-toggle');
    if (expandButton) {
      const initialText = await expandButton.evaluate(el => el.textContent);
      expect(initialText).toContain('Show More');

      // Click to expand
      await expandButton.click();
      
      // Wait for expansion
      await page.waitForTimeout(500);
      
      const expandedText = await expandButton.evaluate(el => el.textContent);
      expect(expandedText).toContain('Show Less');

      // Click to collapse
      await expandButton.click();
      
      // Wait for collapse
      await page.waitForTimeout(500);
      
      const collapsedText = await expandButton.evaluate(el => el.textContent);
      expect(collapsedText).toContain('Show More');
    }
  });

  test('JSON arrays are properly formatted as lists', async () => {
    const analysisContent = await page.$eval('.article-card .article-analysis', el => el.innerHTML);
    
    // Should contain list items for the practical examples array
    expect(analysisContent).toContain('<ul>');
    expect(analysisContent).toContain('<li>Claude Code saves plans to memory</li>');
    expect(analysisContent).toContain('<li>ChatGPT uses embeddings for retrieval</li>');
    expect(analysisContent).toContain('<li>Multi-agent systems use isolation</li>');
  });

  test('JSON keys are properly formatted as readable headers', async () => {
    const analysisContent = await page.$eval('.article-card .article-analysis', el => el.innerHTML);
    
    // Check that camelCase and snake_case keys are converted to readable headers
    expect(analysisContent).toContain('Feynman Technique Breakdown');
    expect(analysisContent).toContain('Four Strategies');
    expect(analysisContent).toContain('Practical Examples');
    
    // Should not contain raw JSON keys
    expect(analysisContent).not.toContain('feynman_technique_breakdown');
    expect(analysisContent).not.toContain('four_strategies');
    expect(analysisContent).not.toContain('practical_examples');
  });

  test('nested objects create proper hierarchy', async () => {
    const analysisContent = await page.$eval('.article-card .article-analysis', el => el.innerHTML);
    
    // Check header hierarchy
    const h3Count = (analysisContent.match(/<h3>/g) || []).length;
    const h4Count = (analysisContent.match(/<h4>/g) || []).length;
    const h5Count = (analysisContent.match(/<h5>/g) || []).length;
    
    // Should have hierarchical structure
    expect(h3Count).toBeGreaterThan(0);
    expect(h4Count).toBeGreaterThan(0);
    expect(h5Count).toBeGreaterThan(0);
  });
});