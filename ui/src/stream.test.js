import { test } from 'node:test';
import assert from 'node:assert/strict';
import { streamAgent } from './stream.js';

test('中文字符和 SSE 边界跨网络分片时仍正确重组', async t => {
  const bytes = new TextEncoder().encode(': heartbeat\n\ndata: {"type":"token","text":"你好"}\n\ndata: {"type":"done","recommendations":[]}\n\n');
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({
    start(controller) {
      for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
      controller.close();
    },
  })));
  const events = [];
  await streamAgent({}, new AbortController().signal, event => events.push(event));
  assert.equal(events[0].text, '你好');
  assert.equal(events[1].type, 'done');
});

test('连接提前结束时不能显示为成功', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('data: {"type":"token","text":"部分回答"}\n\n'));
  await assert.rejects(streamAgent({}, undefined, () => {}), /连接中断/);
});

test('服务端流式错误作为失败反馈', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('data: {"type":"error","text":"执行失败"}\n\n'));
  await assert.rejects(streamAgent({}, undefined, () => {}), /执行失败/);
});
