// 缓冲网络分片，按完整 SSE 帧解析
export async function streamAgent(body, signal, onEvent) {
  const response = await fetch('/api/agent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const error = await response.json();
    const detail = error.detail;
    throw new Error(Array.isArray(detail) ? detail.map(x => x.msg).join('；') : detail || '请求失败');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let completed = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = frame.split('\n').filter(line => line.startsWith('data: '))
          .map(line => line.slice(6)).join('\n');
        if (!data) continue;
        const event = JSON.parse(data);
        if (event.type === 'error') throw new Error(event.text);
        onEvent(event);
        if (event.type === 'done') completed = true;
      }
      if (done) break;
    }
    if (!completed) throw new Error('连接中断，任务未完成，请重新提交');
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
