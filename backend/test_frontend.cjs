// Vueの状態遷移を検証する。ブラウザ描画のテストとは別。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
let definition;
const deadlines = [];
const context = {
    Vue: { createApp(value) { definition = value; return { mount() {} }; } },
    localStorage: { getItem() { return null; }, setItem() {} },
    APP_REQUEST_TIMEOUT_MS: 250000,
    AbortController, clearTimeout,
    setTimeout(callback, delay) { deadlines.push(delay); return setTimeout(callback, delay); },
};
vm.runInNewContext(script, context);
const state = { ...definition.data(), ...definition.methods, $nextTick() {} };
const question = { status: 'question', question: '毎日どのくらい使いますか？',
    answer_type: 'choice', options: ['1時間未満', '1時間以上'], condition: '利用時間', confidence: 30 };
const result = { status: 'result', recommendation: '候補A', conditions: ['未定'],
    reason: '情報が少ないため暫定です', pros: ['利点'], cons: ['欠点'], alternative: '条件次第で候補B', confidence: 35 };
let bodies = [];
context.fetch = async (_url, init) => {
    bodies.push(JSON.parse(init.body));
    return { ok: false, json: async () => ({ error: '再試行してください' }) };
};
(async () => {
    const decimalQuestion = { ...question, question: '1.5時間の移動は可能ですか？', options: ['可能', '難しい'] };
    assert.equal(state.parseResponse(decimalQuestion).question, decimalQuestion.question);
    assert.throws(() => state.parseResponse({ ...question, options: ['同じ', '同じ'] }));
    state.stage = 'question'; state.consultation = '相談'; state.question = question;
    await state.submitAnswer('1時間未満');
    assert.equal(state.history.length, 0);
    assert.equal(state.pending.answer, '1時間未満');
    context.fetch = async (_url, init) => {
        bodies.push(JSON.parse(init.body));
        return { ok: true, json: async () => result };
    };
    await state.sendRequest();
    assert.deepEqual(bodies[0], bodies[1]);
    assert.equal(state.history.length, 1);
    assert.equal(state.confidence, 35);
    assert.equal(state.savedSessions.length, 1);
    state.reset();
    state.stage = 'question'; state.question = question; state.consultation = '相談';
    context.fetch = async (_url, init) => {
        bodies.push(JSON.parse(init.body));
        return { ok: true, json: async () => question };
    };
    await state.skipQuestion();
    assert.equal(state.history.length, 0);
    assert.equal(state.skippedQuestions[0], question.question);
    await state.submitAnswer('わからない・決められない');
    assert.deepEqual(bodies.at(-1).skipped_questions, [question.question]);
    assert.equal(state.history.at(-1).answer, 'わからない・決められない');
    state.reset();
    assert.equal(state.skippedQuestions.length, 0);
    assert.ok(deadlines.length > 0 && deadlines.every(delay => delay === 250000));
    console.log('Frontend state checks passed (retry, history, skips, confidence, reset).');
})().catch(error => { console.error(error); process.exitCode = 1; });
