import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChevronRight, Download, Check, LoaderCircle, Wrench } from 'lucide-react';

const names = {
  inspect_known_data: '查看已知实验', find_candidates: '查询候选序列',
  predict_fitness: '预测适应度', query_mutation_knowledge: '查询突变知识',
  query_position_knowledge: '查询位点知识', compare_residue_properties: '比较氨基酸性质',
  RecommendationBatch: '提交推荐',
};
const components = {
  table: props => <div className="table-scroll"><table {...props} /></div>,
  a: props => <a {...props} target="_blank" rel="noreferrer" />,
};
const mdPlugins = [remarkGfm];

export function RichText({ children }) {
  return <div className="markdown"><Markdown remarkPlugins={mdPlugins} components={components}>{children}</Markdown></div>;
}

// 合并连续文本片段
export function appendEvent(events, event) {
  const last = events.at(-1);
  return event.type === 'token' && last?.type === 'token'
    ? [...events.slice(0, -1), { type: 'token', text: last.text + event.text }]
    : [...events, event];
}

function download(turn, result) {
  const url = URL.createObjectURL(new Blob([JSON.stringify({ task: turn.task, knowledge: turn.knowledge, ...result }, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url; link.download = 'gb1-recommendations.json'; link.click();
  URL.revokeObjectURL(url);
}

function Recommendations({ result, turn }) {
  return <section className="recommendations" aria-label="推荐结果">
    <div className="result-heading"><h3>推荐候选</h3><button className="text-button" onClick={() => download(turn, result)}><Download size={14} />下载 JSON</button></div>
    <p className="muted">预测 log enrichment · 越高越好 · 尚未揭示真实 fitness</p>
    <div className="table-scroll"><table><thead><tr><th>候选</th><th>突变</th><th>预测 fitness</th></tr></thead>
      <tbody>{result.recommendations.map(row => <tr key={row.candidate_id}><td>#{row.candidate_id}</td><td>{row.mutations}</td><td className="number">{row.predicted_fitness.toFixed(3)}</td></tr>)}</tbody></table></div>
    {result.recommendations.map(row => <div className="candidate-detail" key={row.candidate_id}>
      <h4>{row.mutations}</h4><RichText>{row.reason}</RichText>
      <details><summary>完整序列 <ChevronRight size={13} /></summary><code className="sequence">{row.sequence}</code></details>
    </div>)}
  </section>;
}

export default function Conversation({ turns }) {
  return turns.map((turn, index) => {
    const results = new Map(turn.events.filter(e => e.type === 'tool_end').map(e => [e.id, e]));
    return <section className="turn" key={index}>
      <div className="user-message"><p>{turn.task}</p></div>
      <div className="assistant-message">
        <div className="speaker"><span className="assistant-mark">G</span>GB1 <span>科学助手</span></div>
        {turn.events.map((event, i) => {
          if (event.type === 'token') return <RichText key={i}>{event.text}</RichText>;
          if (event.type === 'tool_start') {
            const result = results.get(event.id);
            const running = !result && turn.status === 'running';
            const text = typeof result?.content === 'string' ? result.content : result?.content?.filter(x => x.type === 'text').map(x => x.text).join('\n');
            return <details className="tool" key={i}>
              <summary><ChevronRight className="chevron" size={14} /><Wrench size={13} /><span>{names[event.name] || event.name}</span>
                {running ? <LoaderCircle size={13} className="spin" /> : result?.status === 'success' ? <Check size={13} /> : <small>{result ? '工具报错' : '已停止'}</small>}
              </summary>
              <div className="tool-body"><span className="muted">调用参数</span><pre>{JSON.stringify(event.args, null, 2)}</pre>
                {text && <><span className="muted">工具结果</span><pre>{text}</pre></>}</div>
            </details>;
          }
          if (event.type === 'done' && event.recommendations.length) return <Recommendations key={i} result={event} turn={turn} />;
          if (event.type === 'error') return <p className="error" role="alert" key={i}>{event.text}</p>;
          return null;
        })}
        {turn.status === 'running' && <div className="working"><LoaderCircle size={14} className="spin" /><span>{turn.events.findLast(e => e.type === 'status')?.text || '正在生成回答…'}</span></div>}
        {turn.status === 'stopped' && <p className="turn-note">已停止生成 · 可重新发送问题</p>}
      </div>
    </section>;
  });
}
