/**
 * Graph RAG Agent Factory
 * 
 * Creates a LangChain agent configured for code graph analysis.
 * Supports Azure OpenAI and Google Gemini providers.
 */

import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { SystemMessage } from '@langchain/core/messages';
import { ChatOpenAI, AzureChatOpenAI } from '@langchain/openai';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { ChatAnthropic } from '@langchain/anthropic';
import { ChatOllama } from '@langchain/ollama';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { createGraphRAGTools } from './tools';
import type { 
  ProviderConfig, 
  OpenAIConfig,
  AzureOpenAIConfig, 
  GeminiConfig,
  AnthropicConfig,
  OllamaConfig,
  OpenRouterConfig,
  AgentStreamChunk,
} from './types';
import { 
  type CodebaseContext,
  buildDynamicSystemPrompt,
} from './context-builder';

/**
 * System prompt for the Graph RAG agent
 * 
 * Design principles (based on Aider/Cline research):
 * - Short, punchy directives > long explanations
 * - No template-inducing examples
 * - Let LLM figure out HOW, just tell it WHAT behavior we want
 * - Explicit progress reporting requirement
 * - Anti-laziness directives
 */
/**
 * Base system prompt - exported so it can be used with dynamic context injection
 * 
 * Structure (optimized for instruction following):
 * 1. Identity + GROUNDING mandate (most important)
 * 2. Core protocol (how to work)
 * 3. Tools reference
 * 4. Output format & rules
 * 5. [Dynamic context appended at end]
 */
export const BASE_SYSTEM_PROMPT = `You are Nexus, a Code Analysis Agent with access to a Knowledge Graph. Your responses MUST be grounded.

## ⚠️ MANDATORY: GROUNDING
Every factual claim MUST include a citation.
- File refs: [[src/auth.ts:45-60]] (line range with hyphen)
- NO citation = NO claim. Say "I didn't find evidence" instead of guessing.

## ⚠️ MANDATORY: VALIDATION
Every output MUST be validated.
- Use cypher to validate the results and confirm completeness of context before final output.
- NO validation = NO claim. Say "I didn't find evidence" instead of guessing.
- Do not blindly trust readme or single source of truth. Always validate and cross-reference. Never be lazy.

## 🧠 CORE PROTOCOL
You are an investigator. For each question:
1. **Search** → Use cypher, search or grep to find relevant code
2. **Read** → Use read to see the actual source
3. **Trace** → Use cypher to follow connections in the graph
4. **Cite** → Ground every finding with [[file:line]] or [[Type:Name]]
5. **Validate** → Use cypher to validate the results and confirm completeness of context before final output. ( MUST DO )

## 🛠️ TOOLS
- **\`search\`** — Hybrid search. Results grouped by process with cluster context.
- **\`cypher\`** — Cypher queries against the graph. Use \`{{QUERY_VECTOR}}\` for vector search.
- **\`grep\`** — Regex search. Best for exact strings, TODOs, error codes.
- **\`read\`** — Read file content. Always use after search/grep to see full code.
- **\`explore\`** — Deep dive on a symbol, cluster, or process. Shows membership, participation, connections.
- **\`overview\`** — Codebase map showing all clusters and processes.
- **\`impact\`** — Impact analysis. Shows affected processes, clusters, and risk level.

## 📊 GRAPH SCHEMA
Nodes: File, Folder, Function, Class, Interface, Method, Community, Process
Relations: \`CodeRelation\` with \`type\` property: CONTAINS, DEFINES, IMPORTS, CALLS, EXTENDS, IMPLEMENTS, MEMBER_OF, STEP_IN_PROCESS

## 📐 GRAPH SEMANTICS (Important!)
**Edge Types:**
- \`CALLS\`: Method invocation OR constructor injection. If A receives B as parameter and uses it, A→B is CALLS. This is intentional simplification.
- \`IMPORTS\`: File-level import/include statement.
- \`EXTENDS/IMPLEMENTS\`: Class inheritance.

**Process Nodes:**
- Process labels use format: "EntryPoint → Terminal" (e.g., "onCreate → showToast")
- These are heuristic names from tracing execution flow, NOT application-defined names
- Entry points are detected via export status, naming patterns, and framework conventions

Cypher examples:
- \`MATCH (f:Function) RETURN f.name LIMIT 10\`
- \`MATCH (f:File)-[:CodeRelation {type: 'IMPORTS'}]->(g:File) RETURN f.name, g.name\`

## 📝CRITICAL RULES
- **impact output is trusted.** Do NOT re-validate with cypher. Optionally run the suggested grep commands for dynamic patterns.
- **Cite or retract.** Never state something you can't ground.
- **Read before concluding.** Don't guess from names alone.
- **Retry on failure.** If a tool fails, fix the input and try again.
- **Cyfer tool validation** prefer using cyfer tool in anything that requires graph connections.
- **OUTPUT STYLE** Prefer using tables and mermaid diagrams instead of long explanations.
- ALWAYS USE MERMAID FOR VISUALIZATION AND STRUCTURING THE OUTPUT.

## 🎯 OUTPUT STYLE
Think like a senior architect. Be concise—no fluff, short, precise and to the point.
- Use tables for comparisons/rankings
- Use mermaid diagrams for flows/dependencies
- Surface deep insights: patterns, coupling, design decisions
- End with **TL;DR** (short summary of the response, summing up the response and the most critical parts)

## MERMAID RULES
When generating diagrams:
- NO special characters in node labels: quotes, (), /, &, <, >
- Wrap labels with spaces in quotes: A["My Label"]
- Use simple IDs: A, B, C or auth, db, api
- Flowchart: graph TD or graph LR (not flowchart)
- Always test mentally: would this parse?

BAD:  A[User's Data] --> B(Process & Save)
GOOD: A["User Data"] --> B["Process and Save"]
`;
export const createChatModel = (config: ProviderConfig): BaseChatModel => {
  switch (config.provider) {
    case 'openai': {
      const openaiConfig = config as OpenAIConfig;
      
      if (!openaiConfig.apiKey || openaiConfig.apiKey.trim() === '') {
        throw new Error('OpenAI API key is required but was not provided');
      }
      
      return new ChatOpenAI({
        apiKey: openaiConfig.apiKey,
        modelName: openaiConfig.model,
        temperature: openaiConfig.temperature ?? 0.1,
        maxTokens: openaiConfig.maxTokens,
        configuration: {
          apiKey: openaiConfig.apiKey,
          dangerouslyAllowBrowser: true,
          ...(openaiConfig.baseUrl ? { baseURL: openaiConfig.baseUrl } : {}),
        },
        streaming: true,
      });
    }
    
    case 'azure-openai': {
      const azureConfig = config as AzureOpenAIConfig;

      if (!azureConfig.apiKey || azureConfig.apiKey.trim() === '') {
        throw new Error('Azure OpenAI API key is required but was not provided. Go to Settings → Azure OpenAI and enter your key.');
      }

      return new AzureChatOpenAI({
        azureOpenAIApiKey: azureConfig.apiKey,
        azureOpenAIApiInstanceName: extractInstanceName(azureConfig.endpoint),
        azureOpenAIApiDeploymentName: azureConfig.deploymentName,
        azureOpenAIApiVersion: azureConfig.apiVersion ?? '2024-12-01-preview',
        // Note: gpt-5.2-chat only supports temperature=1 (default)
        streaming: true,
        configuration: {
          dangerouslyAllowBrowser: true,
        },
      });
    }
    
    case 'gemini': {
      const geminiConfig = config as GeminiConfig;

      if (!geminiConfig.apiKey || geminiConfig.apiKey.trim() === '') {
        throw new Error('Google Gemini API key is required but was not provided. Go to Settings → Gemini and enter your key.');
      }

      return new ChatGoogleGenerativeAI({
        apiKey: geminiConfig.apiKey,
        model: geminiConfig.model,
        temperature: geminiConfig.temperature ?? 0.1,
        maxOutputTokens: geminiConfig.maxTokens,
        streaming: true,
      });
    }
    
    case 'anthropic': {
      const anthropicConfig = config as AnthropicConfig;

      if (!anthropicConfig.apiKey || anthropicConfig.apiKey.trim() === '') {
        throw new Error('Anthropic API key is required but was not provided. Go to Settings → Anthropic and enter your key.');
      }

      if (import.meta.env.DEV) {
        const k = anthropicConfig.apiKey;
        console.log(
          `🔑 [Anthropic] key=${k.slice(0, 10)}…${k.slice(-4)} model=${anthropicConfig.model}`
        );
      }

      return new ChatAnthropic({
        anthropicApiKey: anthropicConfig.apiKey,
        model: anthropicConfig.model,
        temperature: anthropicConfig.temperature ?? 0.1,
        maxTokens: anthropicConfig.maxTokens ?? 8192,
        streaming: true,
        // Route through Vite proxy to avoid COEP blocking cross-origin fetch to api.anthropic.com
        anthropicApiUrl: `${typeof self !== 'undefined' && self.location ? self.location.origin : ''}/api/anthropic`,
        // Kill retries — 401s should fail immediately, not retry 6 times silently
        maxRetries: 0,
        clientOptions: {
          dangerouslyAllowBrowser: true,
          maxRetries: 0,
        },
      });
    }
    
    case 'ollama': {
      const ollamaConfig = config as OllamaConfig;
      return new ChatOllama({
        baseUrl: ollamaConfig.baseUrl ?? 'http://localhost:11434',
        model: ollamaConfig.model,
        temperature: ollamaConfig.temperature ?? 0.1,
        streaming: true,
        // Allow longer responses (Ollama default is often 128-2048)
        numPredict: 30000,
        // Increase context window (Ollama default is only 2048!)
        // This is critical for agentic workflows with tool calls
        numCtx: 32768,
      });
    }
    
    case 'openrouter': {
      const openRouterConfig = config as OpenRouterConfig;
      
      // Debug logging
      if (import.meta.env.DEV) {
        console.log('🌐 OpenRouter config:', {
          hasApiKey: !!openRouterConfig.apiKey,
          apiKeyLength: openRouterConfig.apiKey?.length || 0,
          model: openRouterConfig.model,
          baseUrl: openRouterConfig.baseUrl,
        });
      }
      
      if (!openRouterConfig.apiKey || openRouterConfig.apiKey.trim() === '') {
        throw new Error('OpenRouter API key is required but was not provided');
      }
      
      return new ChatOpenAI({
        openAIApiKey: openRouterConfig.apiKey,
        apiKey: openRouterConfig.apiKey, // Fallback for some versions
        modelName: openRouterConfig.model,
        temperature: openRouterConfig.temperature ?? 0.1,
        maxTokens: openRouterConfig.maxTokens,
        configuration: {
          apiKey: openRouterConfig.apiKey, // Ensure client receives it
          baseURL: openRouterConfig.baseUrl ?? 'https://openrouter.ai/api/v1',
          dangerouslyAllowBrowser: true,
        },
        streaming: true,
      });
    }
    
    default:
      throw new Error(`Unsupported provider: ${(config as any).provider}`);
  }
};

/**
 * Extract instance name from Azure endpoint URL
 * e.g., "https://my-resource.openai.azure.com" -> "my-resource"
 */
const extractInstanceName = (endpoint: string): string => {
  try {
    const url = new URL(endpoint);
    const hostname = url.hostname;
    // Extract the first part before .openai.azure.com
    const match = hostname.match(/^([^.]+)\.openai\.azure\.com/);
    if (match) {
      return match[1];
    }
    // Fallback: just use the first part of hostname
    return hostname.split('.')[0];
  } catch {
    return endpoint;
  }
};

/**
 * Create a Graph RAG agent
 */
export const createGraphRAGAgent = (
  config: ProviderConfig,
  executeQuery: (cypher: string) => Promise<any[]>,
  semanticSearch: (query: string, k?: number, maxDistance?: number) => Promise<any[]>,
  semanticSearchWithContext: (query: string, k?: number, hops?: number) => Promise<any[]>,
  hybridSearch: (query: string, k?: number) => Promise<any[]>,
  isEmbeddingReady: () => boolean,
  isBM25Ready: () => boolean,
  fileContents: Map<string, string>,
  codebaseContext?: CodebaseContext
) => {
  const model = createChatModel(config);
  const tools = createGraphRAGTools(
    executeQuery,
    semanticSearch,
    semanticSearchWithContext,
    hybridSearch,
    isEmbeddingReady,
    isBM25Ready,
    fileContents
  );
  
  // Use dynamic prompt if context is provided, otherwise use base prompt
  const systemPrompt = codebaseContext 
    ? buildDynamicSystemPrompt(BASE_SYSTEM_PROMPT, codebaseContext)
    : BASE_SYSTEM_PROMPT;
  
  // Log the full prompt for debugging
  if (import.meta.env.DEV) {
    console.log('🤖 AGENT SYSTEM PROMPT:\n', systemPrompt);
  }
  
  const agent = createReactAgent({
    llm: model as any,
    tools: tools as any,
    messageModifier: new SystemMessage(systemPrompt) as any,
  });
  
  return agent;
};

/**
 * Message type for agent conversation
 */
export interface AgentMessage {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * Stream a response from the agent
 * Uses BOTH streamModes for best of both worlds:
 * - 'values' for state transitions (tool calls, results) in proper order
 * - 'messages' for token-by-token text streaming
 * 
 * This preserves the natural progression: reasoning → tool → reasoning → tool → answer
 */
export async function* streamAgentResponse(
  agent: ReturnType<typeof createReactAgent>,
  messages: AgentMessage[]
): AsyncGenerator<AgentStreamChunk> {
  try {
    const formattedMessages = messages.map(m => ({
      role: m.role,
      content: m.content,
    }));
    
    // Use BOTH modes: 'values' for structure, 'messages' for token streaming
    const stream = await agent.stream(
      { messages: formattedMessages },
      {
        streamMode: ['values', 'messages'] as any,
        // Allow longer tool/reasoning loops (more Cursor-like persistence)
        recursionLimit: 50,
      } as any
    );
    
    // Track what we've yielded to avoid duplicates
    const yieldedToolCalls = new Set<string>();
    const yieldedToolResults = new Set<string>();
    let lastProcessedMsgCount = formattedMessages.length;
    // Track if all tools are done (for distinguishing reasoning vs final content)
    let allToolsDone = true;
    // Track if we've seen any tool calls in this response turn.
    // Anything before the first tool call should be treated as "reasoning/narration"
    // so the UI can show the Cursor-like loop: plan → tool → update → tool → answer.
    let hasSeenToolCallThisTurn = false;
    // Track if messages-mode already delivered text content so we don't double-emit
    // from the values-mode fallback handler.
    // NOTE: createReactAgent uses model.invoke() (non-streaming), so handleLLMNewToken
    // never fires. handleLLMEnd should emit ONE complete message via 'messages' mode
    // as a fallback. If even that fails (e.g. handleChatModelStart threw silently
    // because metadata.langgraph_checkpoint_ns was undefined, or the LLM errored),
    // we fall through to extract the AI message text directly from the values snapshot.
    let hasSentStreamingContent = false;
    // Track which message IDs we've already emitted from the values fallback to
    // avoid re-emitting the same message across multiple values snapshots.
    const yieldedValuesMessageIds = new Set<string>();
    
    for await (const event of stream) {
      // Events come as [streamMode, data] tuples when using multiple modes
      // or just data when using single mode
      let mode: string;
      let data: any;
      
      if (Array.isArray(event) && event.length === 2 && typeof event[0] === 'string') {
        [mode, data] = event;
      } else if (Array.isArray(event) && event[0]?._getType) {
        // Single messages mode format: [message, metadata]
        mode = 'messages';
        data = event;
      } else {
        // Assume values mode
        mode = 'values';
        data = event;
      }
      
      // DEBUG: Enhanced logging
      if (import.meta.env.DEV) {
        if (mode === 'values' && data?.messages) {
          // For values events, show the actual state so we can see if LLM responded
          const msgs: any[] = data.messages ?? [];
          const newMsgs = msgs.slice(lastProcessedMsgCount);
          const summary = newMsgs.map((m: any) => {
            const t = m._getType?.() || m.type || '?';
            const c = typeof m.content === 'string'
              ? m.content.slice(0, 80).replace(/\n/g, '↵')
              : Array.isArray(m.content) ? `[${m.content.length} blocks]` : '';
            const tc = m.tool_calls?.length ? ` tools:${m.tool_calls.length}` : '';
            const err = m.content && typeof m.content === 'string' && m.content.includes('Error') ? ' ⚠️' : '';
            return `${t}:"${c}"${tc}${err}`;
          }).join(' | ');
          console.log(`🔄 [values] total:${msgs.length} new:[${summary || 'none'}]`);
        } else {
          const msgType = mode === 'messages' && data?.[0]?._getType?.() || 'n/a';
          const hasContent = mode === 'messages' && data?.[0]?.content;
          const hasToolCalls = mode === 'messages' && data?.[0]?.tool_calls?.length > 0;
          console.log(`🔄 [${mode}] type:${msgType} content:${!!hasContent} tools:${!!hasToolCalls}`);
        }
      }
      // Handle 'messages' mode - token-by-token streaming
      if (mode === 'messages') {
        const [msg] = Array.isArray(data) ? data : [data];
        if (!msg) continue;
        
        const msgType = msg._getType?.() || msg.type || msg.constructor?.name || 'unknown';
        
        // AIMessageChunk - streaming text tokens
        if (msgType === 'ai' || msgType === 'AIMessage' || msgType === 'AIMessageChunk') {
          const rawContent = msg.content;
          const toolCalls = msg.tool_calls || [];
          
          // Handle content that can be string or array of content blocks
          let content: string = '';
          if (typeof rawContent === 'string') {
            content = rawContent;
          } else if (Array.isArray(rawContent)) {
            // Content blocks format: [{type: 'text', text: '...'}, ...]
            content = rawContent
              .filter((block: any) => block.type === 'text' || typeof block === 'string')
              .map((block: any) => typeof block === 'string' ? block : block.text || '')
              .join('');
          }
          
          // If chunk has content, stream it
          if (content && content.length > 0) {
            // Determine if this is reasoning/narration vs final answer content.
            // - Before the first tool call: treat as reasoning (narration)
            // - Between tool calls/results: treat as reasoning
            // - After all tools are done: treat as final content
            const isReasoning =
              !hasSeenToolCallThisTurn ||
              toolCalls.length > 0 ||
              !allToolsDone;
            hasSentStreamingContent = true; // messages-mode delivered content
            yield {
              type: isReasoning ? 'reasoning' : 'content',
              [isReasoning ? 'reasoning' : 'content']: content,
            };
          }
          
          // Track tool calls from message chunks
          if (toolCalls.length > 0) {
            hasSeenToolCallThisTurn = true;
            allToolsDone = false;
            for (const tc of toolCalls) {
              const toolId = tc.id || `tool-${Date.now()}-${Math.random().toString(36).slice(2)}`;
              if (!yieldedToolCalls.has(toolId)) {
                yieldedToolCalls.add(toolId);
                yield {
                  type: 'tool_call',
                  toolCall: {
                    id: toolId,
                    name: tc.name || tc.function?.name || 'unknown',
                    args: tc.args || (tc.function?.arguments ? JSON.parse(tc.function.arguments) : {}),
                    status: 'running',
                  },
                };
              }
            }
          }
        }
        
        // ToolMessage in messages mode
        if (msgType === 'tool' || msgType === 'ToolMessage') {
          const toolCallId = msg.tool_call_id || '';
          if (toolCallId && !yieldedToolResults.has(toolCallId)) {
            yieldedToolResults.add(toolCallId);
            const result = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
            yield {
              type: 'tool_result',
              toolCall: {
                id: toolCallId,
                name: msg.name || 'tool',
                args: {},
                result: result,
                status: 'completed',
              },
            };
            // After tool result, next AI content could be reasoning or final
            allToolsDone = true;
          }
        }
      }
      
      // Handle 'values' mode - state snapshots for structure
      if (mode === 'values' && data?.messages) {
        const stepMessages = data.messages || [];
        
        // Process new messages for tool calls/results we might have missed,
        // AND fall back to extracting AI text content when messages-mode never fired.
        //
        // WHY THIS FALLBACK EXISTS:
        // createReactAgent calls model.invoke() (non-streaming), so handleLLMNewToken
        // never fires. LangGraph's StreamMessagesHandler.handleLLMEnd() should emit
        // one complete 'messages' event. But if handleChatModelStart threw silently
        // (e.g. metadata.langgraph_checkpoint_ns was absent) or the LLM errored,
        // handleLLMEnd returns early and the response is ONLY in the values snapshot.
        // Without this fallback, the user would see nothing at all.
        for (let i = lastProcessedMsgCount; i < stepMessages.length; i++) {
          const msg = stepMessages[i];
          const msgType = msg._getType?.() || msg.type || 'unknown';
          
          if (msgType === 'ai' || msgType === 'AIMessage') {
            // ── FALLBACK: extract text content from AI message ────────────────
            // Only emit from values mode if messages-mode hasn't already delivered
            // this content. Use the message ID as the dedup key; fall back to the
            // loop index stringified so we never re-emit the same snapshot.
            if (!hasSentStreamingContent) {
              const msgId: string = msg.id ?? `values-msg-${i}`;
              if (!yieldedValuesMessageIds.has(msgId)) {
                yieldedValuesMessageIds.add(msgId);

                const rawContent = msg.content;
                let textContent = '';
                if (typeof rawContent === 'string') {
                  textContent = rawContent;
                } else if (Array.isArray(rawContent)) {
                  textContent = rawContent
                    .filter((b: any) => b.type === 'text' || typeof b === 'string')
                    .map((b: any) => (typeof b === 'string' ? b : b.text || ''))
                    .join('');
                }

                if (textContent.trim()) {
                  if (import.meta.env.DEV) {
                    console.log(`📋 [values-fallback] emitting AI text (${textContent.length} chars) — messages-mode never fired`);
                  }
                  // Determine reasoning vs final: same heuristic as messages mode
                  const isReasoning = !hasSeenToolCallThisTurn || !allToolsDone;
                  yield {
                    type: isReasoning ? 'reasoning' : 'content',
                    [isReasoning ? 'reasoning' : 'content']: textContent,
                  };
                }
              }
            }

            // ── BACKUP: emit tool calls we didn't see in messages mode ────────
            const toolCalls = msg.tool_calls || [];
            for (const tc of toolCalls) {
              const toolId = tc.id || `tool-${Date.now()}`;
              if (!yieldedToolCalls.has(toolId)) {
                allToolsDone = false;
                hasSeenToolCallThisTurn = true;
                yieldedToolCalls.add(toolId);
                yield {
                  type: 'tool_call',
                  toolCall: {
                    id: toolId,
                    name: tc.name || 'unknown',
                    args: tc.args || {},
                    status: 'running',
                  },
                };
              }
            }
          }
          
          // Catch tool results from values mode (backup)
          if (msgType === 'tool' || msgType === 'ToolMessage') {
            const toolCallId = msg.tool_call_id || '';
            if (toolCallId && !yieldedToolResults.has(toolCallId)) {
              yieldedToolResults.add(toolCallId);
              const result = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
              yield {
                type: 'tool_result',
                toolCall: {
                  id: toolCallId,
                  name: msg.name || 'tool',
                  args: {},
                  result: result,
                  status: 'completed',
                },
              };
              allToolsDone = true;
            }
          }
        }
        
        lastProcessedMsgCount = stepMessages.length;
      }
    }
    
    // DEBUG: Stream completed normally
    if (import.meta.env.DEV) {
      console.log('✅ Stream completed normally, yielding done');
    }
    yield { type: 'done' };
  } catch (error) {
    const rawMessage = error instanceof Error ? error.message : String(error);

    // Classify the error and provide actionable guidance
    let userMessage: string;
    const lower = rawMessage.toLowerCase();
    if (lower.includes('authentication') || lower.includes('invalid x-api-key') || lower.includes('401')) {
      userMessage = `Authentication failed — your API key is invalid or expired. Update it in Settings. (${rawMessage})`;
    } else if (lower.includes('not_found') || lower.includes('404') || lower.includes('does not exist')) {
      userMessage = `Model not found — the model name in Settings may be wrong. (${rawMessage})`;
    } else if (lower.includes('rate_limit') || lower.includes('429') || lower.includes('overloaded')) {
      userMessage = `Rate limited — too many requests or the API is overloaded. Wait a moment and retry. (${rawMessage})`;
    } else if (lower.includes('insufficient') || lower.includes('billing') || lower.includes('credit')) {
      userMessage = `Billing issue — check your API account balance / billing settings. (${rawMessage})`;
    } else {
      userMessage = rawMessage;
    }

    // Always log loudly so it's visible even in a web worker console
    console.error('❌ [LLM ERROR]', userMessage);
    if (import.meta.env.DEV) {
      console.error('❌ [LLM ERROR detail]', error);
    }

    yield { 
      type: 'error', 
      error: userMessage,
    };
  }
}

/**
 * Get a non-streaming response from the agent
 * Simpler for cases where streaming isn't needed
 */
/**
 * Verify the API key works by making a minimal request.
 * Call at agent init time — surfaces auth errors immediately instead of
 * letting the user discover them mid-chat when the stream silently hangs.
 */
export const verifyApiConnection = async (
  config: ProviderConfig
): Promise<{ ok: boolean; error?: string }> => {
  if (config.provider !== 'anthropic') return { ok: true }; // only anthropic uses proxy

  const anthropicConfig = config as AnthropicConfig;
  const baseUrl = typeof self !== 'undefined' && self.location
    ? self.location.origin
    : '';
  const url = `${baseUrl}/api/anthropic/v1/messages`;

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': anthropicConfig.apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model: anthropicConfig.model,
        max_tokens: 1,
        messages: [{ role: 'user', content: 'ping' }],
      }),
    });

    if (res.ok) {
      console.log('✅ [API KEY VERIFIED] Anthropic connection OK');
      // consume body to avoid connection leak
      await res.text().catch(() => {});
      return { ok: true };
    }

    const body = await res.text().catch(() => '');
    if (res.status === 401) {
      const msg = `❌ [INVALID API KEY] Anthropic returned 401. Your key (${anthropicConfig.apiKey.slice(0, 12)}…) is invalid or expired. Update it in Settings → Anthropic.`;
      console.error(msg);
      console.error('   Raw response:', body);
      return { ok: false, error: msg };
    }

    // 400 = bad model name, but key is valid
    if (res.status === 400) {
      if (body.includes('model')) {
        const msg = `⚠️ [BAD MODEL NAME] Key is valid but model "${anthropicConfig.model}" was rejected. Check Settings → Anthropic → Model.`;
        console.warn(msg);
        return { ok: false, error: msg };
      }
    }

    // Other errors (429, 500, etc.) — key might be OK
    console.warn(`⚠️ [API CHECK] Anthropic returned ${res.status}: ${body.slice(0, 200)}`);
    return { ok: true }; // don't block on transient errors
  } catch (err) {
    const msg = `⚠️ [API CHECK] Network error — proxy may not be running: ${err}`;
    console.warn(msg);
    return { ok: false, error: msg };
  }
};

export const invokeAgent = async (
  agent: ReturnType<typeof createReactAgent>,
  messages: AgentMessage[]
): Promise<string> => {
  const formattedMessages = messages.map(m => ({
    role: m.role,
    content: m.content,
  }));
  
  const result = await agent.invoke({ messages: formattedMessages });
  
  // result.messages is the full conversation state
  const lastMessage = result.messages[result.messages.length - 1];
  return lastMessage?.content?.toString() ?? 'No response generated.';
};

