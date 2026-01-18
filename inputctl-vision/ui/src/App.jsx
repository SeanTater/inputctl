import { clsx } from 'clsx';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity,
  CheckCircle,
  Eye,
  Image as ImageIcon,
  List,
  MousePointer2,
  Terminal
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export default function App() {
  const [state, setState] = useState(null);
  const [selectedIteration, setSelectedIteration] = useState(0);
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const [injectionText, setInjectionText] = useState('');
  const socketRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [state?.iterations]);

  const togglePause = async () => {
    const endpoint = paused ? '/api/resume' : '/api/pause';
    await fetch(endpoint, { method: 'POST' });
    setPaused(!paused);
  };

  const handleInject = async () => {
    if (!injectionText.trim()) return;
    await fetch('/api/inject', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        role: 'user',
        content: injectionText,
      }),
    });
    setInjectionText('');
  };

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws`;

    const connect = () => {
      console.log('Connecting to', wsUrl);
      const ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        console.log('Connected to debugger');
      };

      ws.onmessage = (event) => {
        const newState = JSON.parse(event.data);
        setState(newState);

        // Update local paused state from server if it matches a convention
        // (We didn't explicitly add it to AgentState but we could)

        if (newState.iterations?.length > 0) {
          // Auto-select latest iteration if we were at the end
          setSelectedIteration(prev => {
            if (prev === (state?.iterations?.length || 0) - 1) {
              return newState.iterations.length - 1;
            }
            return prev;
          });
        }
      };

      ws.onclose = () => {
        setConnected(false);
        console.log('Disconnected, retrying in 2s...');
        setTimeout(connect, 2000);
      };
    };

    connect();
    return () => socketRef.current?.close();
  }, []);

  if (!state) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950 text-slate-200">
        <motion.div
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="flex flex-col items-center gap-4"
        >
          <Activity className="w-12 h-12 text-indigo-500" />
          <p className="text-xl font-light tracking-widest">AWAITING AGENT...</p>
        </motion.div>
      </div>
    );
  }

  const currentIter = state.iterations[selectedIteration];

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-200 font-sans overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div className={cn(
            "w-3 h-3 rounded-full animate-pulse",
            connected ? "bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]" : "bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.5)]"
          )} />
          <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
            VISIONCTL <span className="text-slate-500 font-light">DEBUGGER</span>
          </h1>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={togglePause}
            className={cn(
              "flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold transition-all border",
              paused
                ? "bg-amber-500/10 border-amber-500/50 text-amber-500 hover:bg-amber-500/20"
                : "bg-indigo-500/10 border-indigo-500/50 text-indigo-400 hover:bg-indigo-500/20"
            )}
          >
            {paused ? (
              <>
                <div className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
                RESUME AGENT
              </>
            ) : (
              <>
                <div className="w-2 h-2 rounded-full bg-indigo-500" />
                PAUSE AGENT
              </>
            )}
          </button>

          <div className="flex items-center gap-6 border-l border-slate-800 pl-4">
            <div className="flex flex-col items-end">
              <span className="text-[10px] uppercase tracking-widest text-slate-500">Goal</span>
              <span className="text-sm font-medium text-indigo-300 max-w-md truncate">{state.goal}</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800 border border-slate-700">
              <span className="text-xs font-semibold uppercase tracking-tighter text-slate-400">Status</span>
              <span className={cn(
                "text-xs font-bold",
                state.status === 'Running' ? "text-indigo-400" :
                  state.status === 'Success' ? "text-emerald-400" : "text-rose-400"
              )}>{state.status}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="flex flex-1 overflow-hidden">
        {/* Sidebar: Iterations */}
        <div className="w-64 border-r border-slate-800 bg-slate-900/30 overflow-y-auto">
          <div className="p-4 border-b border-slate-800 flex items-center justify-between sticky top-0 bg-slate-950 z-10">
            <span className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">Iterations</span>
            <span className="text-xs text-slate-400">{state.iterations.length} total</span>
          </div>
          <div className="flex flex-col p-2 gap-1">
            {state.iterations.map((iter, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedIteration(idx)}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-all group",
                  selectedIteration === idx
                    ? "bg-indigo-600/20 border border-indigo-500/50 text-indigo-200"
                    : "hover:bg-slate-800 text-slate-400 border border-transparent"
                )}
              >
                <div className={cn(
                  "w-6 h-6 rounded flex items-center justify-center text-[10px] font-bold",
                  selectedIteration === idx ? "bg-indigo-500 text-white" : "bg-slate-800 text-slate-500 group-hover:bg-slate-700 group-hover:text-slate-300"
                )}>
                  {idx + 1}
                </div>
                <div className="flex flex-col flex-1 overflow-hidden">
                  <span className="text-sm font-medium truncate capitalize">
                    {iter.tool_calls?.[0]?.name || 'Analyzing'}
                  </span>
                  <span className="text-[10px] opacity-50">
                    {new Date(iter.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Content Area */}
        {currentIter ? (
          <div className="flex-1 flex flex-col overflow-hidden bg-slate-950">
            <div className="grid grid-cols-12 h-full overflow-hidden">
              {/* Screenshot View */}
              <div className="col-span-12 lg:col-span-7 xl:col-span-8 p-6 flex flex-col gap-4 overflow-hidden border-r border-slate-800">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-slate-400 flex items-center gap-2 uppercase tracking-wide">
                    <ImageIcon className="w-4 h-4" /> Visual State
                  </h2>
                  <div className="flex items-center gap-4 text-xs text-slate-500">
                    <div className="flex items-center gap-2">
                      <Eye className="w-3 h-3" />
                      {currentIter.viewport ? `Viewport: ${currentIter.viewport.width}x${currentIter.viewport.height}` : 'Full Screen'}
                    </div>
                  </div>
                </div>

                <div className="flex-1 flex flex-col gap-4 overflow-hidden">
                  {/* Context View */}
                  <div className="flex-1 relative flex flex-col min-h-0 bg-slate-900/50 rounded-xl border border-slate-800 p-2 overflow-hidden">
                    <div className="flex items-center justify-between mb-2">
                      <div className="px-2 py-0.5 rounded bg-black/60 backdrop-blur-md border border-white/10 text-[10px] uppercase font-bold text-slate-400">
                        Context (Desktop)
                      </div>
                    </div>

                    <div className="flex-1 relative flex items-center justify-center overflow-hidden">
                      <div
                        className="relative shadow-2xl"
                        style={{
                          aspectRatio: `${state.screen_width} / ${state.screen_height}`,
                          maxHeight: '100%',
                          maxWidth: '100%'
                        }}
                      >
                        {currentIter.screenshot_b64 ? (
                          <img
                            src={`data:image/png;base64,${currentIter.screenshot_b64}`}
                            className="w-full h-full block"
                            alt="Desktop Context"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-slate-700 bg-black font-mono text-[10px]">
                            {currentIter.model_screenshot_b64 ? "USING MODEL VIEW AS CONTEXT" : "NO CONTEXT"}
                          </div>
                        )}

                        {/* Viewport Highlight on Context */}
                        {currentIter.viewport && currentIter.screenshot_b64 && (
                          <div
                            style={{
                              left: `${(currentIter.viewport.x / state.screen_width) * 100}%`,
                              top: `${(currentIter.viewport.y / state.screen_height) * 100}%`,
                              width: `${(currentIter.viewport.width / state.screen_width) * 100}%`,
                              height: `${(currentIter.viewport.height / state.screen_height) * 100}%`,
                            }}
                            className="absolute border-2 border-indigo-500/50 bg-indigo-500/10 pointer-events-none"
                          >
                            <div className="absolute -top-5 left-0 text-[10px] font-bold text-indigo-400 px-1 rounded bg-slate-900 border border-indigo-500/50 whitespace-nowrap">
                              ACTIVE VIEWPORT
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Model Input View */}
                  <div className="flex-1 relative flex flex-col min-h-0 bg-slate-900/50 rounded-xl border border-slate-800 p-2 overflow-hidden">
                    <div className="flex items-center justify-between mb-2">
                      <div className="px-2 py-0.5 rounded bg-indigo-500/80 backdrop-blur-md border border-white/10 text-[10px] uppercase font-bold text-white">
                        Model Input (What the LLM sees)
                      </div>
                    </div>

                    <div className="flex-1 relative flex items-center justify-center overflow-hidden">
                      <div
                        className="relative shadow-2xl bg-black"
                        style={{
                          aspectRatio: currentIter.viewport
                            ? `${currentIter.viewport.width} / ${currentIter.viewport.height}`
                            : `${state.screen_width} / ${state.screen_height}`,
                          maxHeight: '100%',
                          maxWidth: '100%'
                        }}
                      >
                        {currentIter.model_screenshot_b64 ? (
                          <img
                            src={`data:image/png;base64,${currentIter.model_screenshot_b64}`}
                            className="w-full h-full block"
                            alt="Model Input"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-slate-700 font-mono text-[10px]">
                            NO MODEL INPUT
                          </div>
                        )}

                        {/* Visualizer Overlays on Model Input */}
                        <AnimatePresence>
                          {currentIter.tool_calls.map((call, i) => (
                            (call.name === 'move_to' || call.name === 'click') && (
                              <motion.div
                                initial={{ scale: 0, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                key={i}
                                style={{
                                  left: `${(call.arguments.x / 10)}%`,
                                  top: `${(call.arguments.y / 10)}%`
                                }}
                                className="absolute -translate-x-1/2 -translate-y-1/2 z-20 pointer-events-none"
                              >
                                <div className="relative">
                                  <motion.div
                                    animate={{ scale: [1, 2, 1], opacity: [0.5, 0, 0.5] }}
                                    transition={{ repeat: Infinity, duration: 2 }}
                                    className="absolute inset-0 w-8 h-8 -left-4 -top-4 rounded-full bg-rose-500"
                                  />
                                  <MousePointer2 className="w-6 h-6 text-white drop-shadow-[0_0_8px_rgba(0,0,0,1)]" />
                                </div>
                              </motion.div>
                            )
                          ))}
                        </AnimatePresence>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Sidebar: Details */}
              <div className="col-span-12 lg:col-span-5 xl:col-span-4 flex flex-col overflow-hidden">
                <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-8 custom-scrollbar">
                  {/* Messages Section */}
                  <section>
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                      <Terminal className="w-3 h-3" /> Conversation
                    </h3>
                    <div className="flex flex-col gap-3">
                      {currentIter.messages.filter(m => m.role !== 'system').map((msg, idx) => (
                        <div key={idx} className={cn(
                          "p-3 rounded-lg text-xs leading-relaxed border",
                          msg.role === 'user' ? "bg-slate-900 border-slate-800" : "bg-indigo-900/10 border-indigo-500/20"
                        )}>
                          <div className="flex items-center gap-2 mb-1">
                            <span className={cn(
                              "text-[10px] font-bold uppercase",
                              msg.role === 'user' ? "text-slate-500" : "text-indigo-400"
                            )}>{msg.role}</span>
                          </div>
                          <div className="text-slate-300 font-mono whitespace-pre-wrap break-words">
                            {msg.content}
                          </div>
                        </div>
                      ))}
                      <div ref={messagesEndRef} />
                    </div>

                    {/* Message Injection */}
                    <div className="mt-4 pt-4 border-t border-slate-800">
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          value={injectionText}
                          onChange={(e) => setInjectionText(e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleInject()}
                          placeholder="Guide the agent..."
                          className="flex-1 bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-indigo-500 transition-colors"
                        />
                        <button
                          onClick={handleInject}
                          disabled={!injectionText.trim()}
                          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600 p-2 rounded-lg transition-colors"
                        >
                          <CheckCircle className="w-4 h-4" />
                        </button>
                      </div>
                      <p className="text-[10px] text-slate-500 mt-2 px-1 italic">
                        Injected messages will be added to the agent's history in the next iteration.
                      </p>
                    </div>
                  </section>

                  {/* Tool Calls Section */}
                  <section>
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                      <List className="w-3 h-3" /> Tool Executions
                    </h3>
                    <div className="flex flex-col gap-4">
                      {currentIter.tool_calls.map((call, idx) => (
                        <div key={idx} className="flex flex-col gap-2 p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-lg">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-bold text-indigo-400 italic">
                              {call.name}()
                            </span>
                            {call.response ? (
                              <CheckCircle className="w-4 h-4 text-emerald-500" />
                            ) : (
                              <div className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
                            )}
                          </div>

                          <div className="flex flex-col gap-2">
                            <div className="text-[10px] uppercase tracking-tighter text-slate-500 font-bold">Arguments</div>
                            <pre className="p-2 rounded bg-black/40 text-[11px] text-slate-400 overflow-x-auto border border-white/5">
                              {JSON.stringify(call.arguments, null, 2)}
                            </pre>
                          </div>

                          {call.response && (
                            <div className="flex flex-col gap-2 mt-2 pt-2 border-t border-slate-800">
                              <div className="text-[10px] uppercase tracking-tighter text-slate-500 font-bold">Response</div>
                              <pre className="p-2 rounded bg-emerald-950/20 text-[11px] text-emerald-200/70 overflow-x-auto border border-emerald-500/10">
                                {JSON.stringify(call.response, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      ))}
                      {currentIter.tool_calls.length === 0 && (
                        <div className="text-sm text-slate-600 text-center py-8 border-2 border-dashed border-slate-800 rounded-xl">
                          Agent is thinking...
                        </div>
                      )}
                    </div>
                  </section>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-600">
            Select an iteration to see details
          </div>
        )}
      </main>

      {/* CSS for custom scrollbar */}
      <style dangerouslySetInnerHTML={{
        __html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #1e293b;
          border-radius: 20px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #334155;
        }
      `}} />
    </div>
  );
}
