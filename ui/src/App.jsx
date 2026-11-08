import { useEffect, useRef, useState } from 'react';
import { ArrowUp, Square, Plus, MessageSquare, PanelLeftClose, PanelLeftOpen, ArrowDown, Trash2, Dna, ListChecks } from 'lucide-react';
import Conversation, { appendEvent } from './Conversation.jsx';
import { streamAgent } from './stream.js';

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error((await response.json()).detail || '请求失败');
  return response.json();
}

export default function App() {
  const [chats, setChats] = useState([]);
  const [chatId, setChatId] = useState(null);
  const [turns, setTurns] = useState([]);
  const [task, setTask] = useState('');
  const [sequence, setSequence] = useState('');
  const [knowledge, setKnowledge] = useState(false);
  const [recommend, setRecommend] = useState(false);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [sidebar, setSidebar] = useState(() => window.innerWidth > 760);
  const [atBottom, setAtBottom] = useState(true);
  const controller = useRef(null);
  const scroller = useRef(null);
  const input = useRef(null);
  const follow = useRef(true);

  async function refreshChats() { setChats(await api('/api/conversations')); }
  useEffect(() => {
    Promise.all([api('/api/config'), api('/api/conversations')]).then(async ([config, rows]) => {
      setSequence(config.sequence); setChats(rows);
      const previous = localStorage.getItem('gb1-current-chat');
      if (rows.some(row => row.id === previous)) {
        const chat = await api(`/api/conversations/${previous}`);
        setChatId(previous); setTurns(chat.turns);
      }
    }).catch(() => setError('无法连接服务，请启动后端后刷新页面')).finally(() => setLoading(false));
    return () => controller.current?.abort();
  }, []);

  useEffect(() => {
    if (follow.current && scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight;
  }, [turns]);
  useEffect(() => {
    if (input.current) {
      input.current.style.height = 'auto';
      input.current.style.height = `${Math.min(input.current.scrollHeight, 160)}px`;
    }
  }, [task]);

  function newChat() {
    if (busy || loading) return;
    setChatId(null); setTurns([]); setTask(''); setError(''); follow.current = true; setAtBottom(true);
    localStorage.removeItem('gb1-current-chat');
    if (window.innerWidth <= 760) setSidebar(false);
    input.current?.focus();
  }

  async function openChat(id) {
    setLoading(true); setError('');
    try {
      const chat = await api(`/api/conversations/${id}`);
      follow.current = true; setAtBottom(true); setChatId(id); setTurns(chat.turns); setTask('');
      localStorage.setItem('gb1-current-chat', id);
      if (window.innerWidth <= 760) setSidebar(false);
    } catch (error) { setError(error.message); }
    finally { setLoading(false); }
  }

  async function deleteChat() {
    if (!chatId || !window.confirm('删除这段会话及其聊天上下文？已保存的推荐报告会保留。')) return;
    try {
      await api(`/api/conversations/${chatId}`, { method: 'DELETE' });
      newChat(); await refreshChats();
    } catch (error) { setError(error.message); }
  }

  async function submit(event) {
    event.preventDefault();
    if (busy || loading || !task.trim() || !sequence) return;
    const id = chatId || crypto.randomUUID();
    const request = { conversation_id: id, task: task.trim(), sequence, knowledge, recommend };
    setChatId(id); setTask(''); setError(''); setBusy(true); follow.current = true; setAtBottom(true);
    localStorage.setItem('gb1-current-chat', id);
    setTurns(current => [...current, { task: request.task, knowledge, recommend, events: [], status: 'running' }]);
    if (!chatId) setChats(current => [{ id, title: request.task.slice(0, 36) }, ...current]);
    controller.current = new AbortController();
    // 仅更新当前回复
    const update = fn => setTurns(current => [...current.slice(0, -1), fn(current.at(-1))]);
    try {
      await streamAgent(request, controller.current.signal, event => {
        update(turn => ({ ...turn, events: appendEvent(turn.events, event), status: event.type === 'done' ? 'complete' : turn.status }));
      });
    } catch (error) {
      const stopped = error.name === 'AbortError';
      update(turn => ({ ...turn, status: stopped ? 'stopped' : 'error', events: stopped ? turn.events : appendEvent(turn.events, { type: 'error', text: error.message }) }));
    } finally {
      setBusy(false);
      refreshChats().catch(() => setError('会话列表更新失败，可刷新页面重试'));
      input.current?.focus();
    }
  }

  const title = chats.find(chat => chat.id === chatId)?.title || '新对话';
  return <div className={`app ${sidebar ? 'sidebar-open' : ''}`}>
    {sidebar && <button className="sidebar-backdrop" aria-label="关闭侧栏" onClick={() => setSidebar(false)} />}
    <aside className="sidebar" aria-label="会话侧栏" inert={!sidebar}>
      <div className="brand"><Dna size={24} /><span>GB1 Lab</span><button className="icon-button" aria-label="收起侧栏" onClick={() => setSidebar(false)}><PanelLeftClose size={18} /></button></div>
      <button className="new-chat" onClick={newChat} disabled={busy || loading}><Plus size={19} />新对话</button>
      <div className="history-label">对话记录</div>
      <nav className="history">{chats.map(chat => <button key={chat.id} disabled={busy || loading} className={`chat-link ${chat.id === chatId ? 'selected' : ''}`} onClick={() => openChat(chat.id)} title={chat.title}><MessageSquare size={15} /><span>{chat.title}</span></button>)}
        {!chats.length && <p className="history-empty">你的对话将保存在这里</p>}</nav>
    </aside>
    <main className="chat-shell">
      <header className="chat-header">
        {!sidebar && <button className="icon-button" aria-label="展开侧栏" onClick={() => setSidebar(true)}><PanelLeftOpen size={19} /></button>}
        <div className="heading"><h1>{title}</h1><span>自训练 Transformer <span className="header-dot">·</span> MCP</span></div>
        {chatId && <button className="icon-button" aria-label="删除当前会话" disabled={busy || loading} onClick={deleteChat}><Trash2 size={17} /></button>}
      </header>
      <div className="conversation-scroll" ref={scroller} onScroll={event => {
        const el = event.currentTarget;
        follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 70;
        setAtBottom(follow.current);
      }}>
        <div className={`conversation ${turns.length ? '' : 'is-empty'}`}>
          {!turns.length && <div className="welcome"><span className="welcome-label">GB1 / SCIENTIFIC ASSISTANT</span><h2>下一步，探索什么？</h2><p>分析实验，比较突变，寻找值得验证的候选。</p>
          </div>}
          <Conversation turns={turns} />
          {loading && <p className="muted">正在读取对话…</p>}
        </div>
      </div>
      <div className="composer-area">
        {!atBottom && <button className="to-bottom" aria-label="回到最新消息" onClick={() => { follow.current = true; scroller.current.scrollTop = scroller.current.scrollHeight; setAtBottom(true); }}><ArrowDown size={16} /></button>}
        {error && <p className="error global-error" role="alert">{error}</p>}
        <form className="composer" onSubmit={submit}>
          <div className="wild-type-input">
            <div className="sequence-heading"><label htmlFor="sequence"><Dna size={14} />GB1 野生型</label><span>{sequence.replace(/\s/g, '').length} aa</span></div>
            <textarea id="sequence" aria-label="GB1 野生型序列" value={sequence} onChange={e => setSequence(e.target.value)} disabled={busy} rows={2} spellCheck={false} autoCapitalize="characters" placeholder="输入 GB1 野生型氨基酸序列" />
          </div>
          <div className="question-input">
          <textarea ref={input} id="task" aria-label="发送消息" placeholder={turns.length ? '继续追问，或提出新的假设…' : '向 GB1 科学助手提问…'} value={task} onChange={e => setTask(e.target.value)} rows={1}
            onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); event.currentTarget.form.requestSubmit(); } }} />
          <div className="composer-toolbar"><div className="composer-options">
            <button type="button" className="option-button" aria-pressed={knowledge} disabled={busy} onClick={() => setKnowledge(value => !value)}><Dna size={15} />知识增强</button>
            <button type="button" className="option-button" aria-pressed={recommend} disabled={busy} onClick={() => setRecommend(value => !value)} title="开启后生成 Top-10 推荐，关闭时进行对话分析"><ListChecks size={15} />推荐 Top-10</button>
          </div>
          {busy ? <button className="send-button stop" type="button" aria-label="停止生成" onClick={() => controller.current?.abort()}><Square size={15} fill="currentColor" /></button>
            : <button className="send-button" type="submit" aria-label="发送" disabled={!task.trim() || !sequence || loading}><ArrowUp size={20} /></button>}</div>
          </div>
        </form>
        <p className="composer-note">GB1 的预测分数不等于实验测量值，请结合证据判断。<span role="status">{busy ? '正在生成' : '就绪'}</span></p>
      </div>
    </main>
  </div>;
}
