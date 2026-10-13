import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from 'file:///C:/Users/surya/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const root = 'C:/Users/surya/Desktop/Project Dump/Causal-Discovery-Edge-IIoT-Optimization';
const skill = 'C:/Users/surya/.codex/plugins/cache/openai-primary-runtime/presentations/26.1009.11318/skills/presentations';
const build = path.join(root, 'presentations/.build');
const output = path.join(root, 'presentations/output');
const finalPath = path.join(output, 'edge_iiot_project_review_phases1_5_v5.pptx');
const draftPath = path.join(build, 'draft.pptx');
const font = 'Arial';
const C = { ink: '#17282D', muted: '#526168', teal: '#11666B', red: '#A23C27', gray: '#D3DADB', bg: '#FAFBFA' };

const pres = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const imageFiles = [
  'presentations/assets/nitt-emblem.png',
  'reports/phase2/final_dag.png',
  'reports/phase3/training_curves.png',
  'reports/phase4_v3/reward_curve.png',
];
const imageBytes = new Map(await Promise.all(imageFiles.map(async file => [
  file, new Uint8Array(await fs.readFile(path.join(root, file))),
])));

function text(slide, content, x, y, w, h, size = 22, opts = {}) {
  const shape = slide.shapes.add({
    geometry: 'textbox',
    position: { left: x, top: y, width: w, height: h },
    fill: 'none', line: { style: 'solid', fill: 'none', width: 0 },
  });
  shape.text = content;
  shape.text.style = {
    typeface: font, fontSize: size, bold: Boolean(opts.bold),
    color: opts.color || C.ink, alignment: opts.align || 'left',
    verticalAlignment: 'middle', autoFit: 'shrinkText',
    wrap: 'square', insets: 0,
  };
  return shape;
}

function line(slide, x, y, w, color = C.gray, width = 1) {
  return slide.shapes.add({ geometry: 'line', position: { left: x, top: y, width: w, height: 0 },
    fill: 'none', line: { style: 'solid', fill: color, width } });
}

function base(title, number, label = 'PROJECT REVIEW') {
  const slide = pres.slides.add();
  slide.background.fill = C.bg;
  text(slide, label, 70, 25, 720, 26, 15, { bold: true, color: C.red });
  text(slide, title, 70, 61, 1135, 64, 37, { bold: true });
  line(slide, 70, 135, 1135, C.gray, 1.3);
  text(slide, `NIT Tiruchirappalli  |  ${String(number).padStart(2, '0')}`, 70, 672, 500, 22, 14, { color: C.muted });
  return slide;
}

function note(slide, content) { slide.speakerNotes.text = content; }
function image(slide, file, x, y, w, h, alt) {
  slide.images.add({ blob: imageBytes.get(file), contentType: 'image/png', alt, fit: 'contain',
    position: { left: x, top: y, width: w, height: h } });
}

// 1. Cover
{
  const s = pres.slides.add(); s.background.fill = C.bg;
  image(s, 'presentations/assets/nitt-emblem.png', 990, 50, 205, 205, 'Official NIT Tiruchirappalli emblem');
  text(s, 'National Institute of Technology\nTiruchirappalli', 72, 54, 805, 76, 23, { color: C.muted });
  line(s, 72, 170, 760, C.red, 3);
  text(s, 'Causality-Aware GNN-RL\nfor Edge-IIoT Optimization', 72, 216, 1040, 175, 54, { bold: true });
  text(s, 'Project review  |  Phases 1–5', 72, 422, 800, 38, 25, { color: C.teal });
  text(s, '[Your Name]   •   [Roll Number]', 72, 554, 900, 44, 26, { bold: true });
  note(s, 'Present your name and roll number after replacing the placeholders. Official emblem source: https://www.nitt.edu/home/about/High-Resolution-Emblem.png . The emblem belongs to NIT Tiruchirappalli and is shown without alteration.');
}

// 2. Problem statement: context, gap, objective and testable question
{
  const s = base('Problem statement', 2, 'RESEARCH MOTIVATION');
  text(s, 'Context', 76, 170, 210, 38, 22, { bold: true, color: C.teal });
  text(s, 'Edge-IIoT traffic and communication patterns change over time, shifting the load across connected devices and servers.', 285, 168, 885, 76, 23);
  line(s, 76, 252, 1090);
  text(s, 'Research gap', 76, 270, 210, 38, 22, { bold: true, color: C.red });
  text(s, 'Traffic datasets record what happened, but not the latency, energy or delivery outcomes of routing actions that were never taken.', 285, 268, 885, 88, 23);
  line(s, 76, 368, 1090);
  text(s, 'Objective', 76, 386, 210, 38, 22, { bold: true, color: C.teal });
  text(s, 'Represent the network as a graph, learn a routing policy, and screen candidate actions with a structural model before execution.', 285, 384, 885, 89, 23);
  line(s, 76, 486, 1090);
  text(s, 'Research question', 76, 506, 1100, 37, 22, { bold: true, color: C.red });
  text(s, 'Can graph-aware learning and counterfactual screening improve routing performance and explainability under changing traffic?', 76, 548, 1090, 91, 27, { bold: true });
  note(s, 'This is the problem statement derived from the user-supplied project brief. The dataset has no recorded outcomes for the six alternative routing actions, so this implementation evaluates decisions in a synthetic three-server simulator. Latency and energy are proxies. The proposed learned action-aware SCM remains a research aspiration; the current Phase 5 screen is an exact SCM of the simulator.');
}

// 3. Scope and method choices, separated from the problem
{
  const s = base('Scope and proposed method', 3, 'RESEARCH DESIGN');
  text(s, 'Network data', 76, 169, 235, 35, 22, { bold: true, color: C.teal });
  text(s, 'Edge-IIoTset (primary) or TON_IoT; graph nodes represent devices or edge servers, with communication/dependency links.', 316, 166, 852, 74, 21);
  line(s, 76, 252, 1090);
  text(s, 'Inference', 76, 268, 235, 35, 22, { bold: true, color: C.teal });
  text(s, 'NOTEARS, DirectLiNGAM or PC for a candidate causal structure; refine using domain knowledge.', 316, 266, 852, 72, 21);
  line(s, 76, 350, 1090);
  text(s, 'Decision model', 76, 366, 235, 35, 22, { bold: true, color: C.teal });
  text(s, 'GNN graph embedding with a PPO or DQN agent; evaluate alternative actions with an SCM before execution.', 316, 364, 852, 74, 21);
  line(s, 76, 448, 1090);
  text(s, 'Evidence', 76, 464, 235, 35, 22, { bold: true, color: C.teal });
  text(s, 'Latency, energy, throughput, PDR, task completion, reward, distribution-shift robustness and explanation fidelity.', 316, 462, 852, 76, 21);
  line(s, 76, 558, 1090, C.red, 1.5);
  text(s, 'Implemented variant: Edge-IIoTset, observed endpoints, NOTEARS, GATv2, DQN and exact simulator SCM.', 76, 578, 1090, 56, 20, { bold: true, color: C.red });
  note(s, 'This slide records the supplied problem scope separately from the actual implemented choices. TON_IoT, DirectLiNGAM, PC and PPO are alternatives, not runs in this repository. Observed IP endpoints are used as graph nodes without validated device/server roles. The Phase 2 learned linear SCM is observational. The Phase 5 action-aware SCM is an independent exact model of the simulator; it is not learned from the Phase 2 DAG.');
}

// 4. Required all-six overview
{
  const s = base('Six-phase research path', 4);
  const rows = [
    ['01', 'Traffic to graphs', 'Directed 60-second PyG snapshots'],
    ['02', 'Causal discovery', 'Nine graph summaries, refined candidate DAG'],
    ['03', 'Graph representation', 'Bidirectional edge-aware GATv2, 64D embedding'],
    ['04', 'Routing control', 'DQN over six actions in a three-server simulator'],
    ['05', 'Counterfactual safety', 'Screen all six actions with exact simulator equations'],
    ['06', 'Final evaluation', 'Ablations, workload shifts, and explanation fidelity'],
  ];
  rows.forEach((r, i) => {
    const y = 174 + i * 77;
    text(s, r[0], 76, y, 60, 49, 28, { bold: true, color: i === 5 ? C.muted : C.red });
    text(s, r[1], 155, y, 330, 49, 25, { bold: true, color: i === 5 ? C.muted : C.ink });
    text(s, r[2], 500, y, 690, 49, 21, { color: i === 5 ? C.muted : C.ink });
    if (i < 5) line(s, 76, y + 64, 1110);
  });
  note(s, 'This review presents the implementation through Phase 5. Phase 6 is shown as the final evaluation stage, but its results are outside the scope of this deck. Phases 1 to 5 have standalone artifacts.');
}

// 5. Editable data-contract diagrams
{
  const s = base('Input and output structure', 5, 'DATA CONTRACT');
  text(s, 'Input data', 76, 178, 495, 40, 27, { bold: true, color: C.teal });
  text(s, 'Packet record', 76, 234, 215, 33, 22, { bold: true });
  text(s, 'time  |  source  |  destination  |  traffic features', 76, 272, 510, 52, 19);
  line(s, 76, 339, 505);
  text(s, 'Directed 60-second graph', 76, 357, 480, 34, 22, { bold: true });
  text(s, 'nodes x [N, 7]    edges [2, E]\nedge attributes [E, 66]', 76, 399, 510, 70, 21);
  line(s, 76, 490, 505);
  text(s, 'Window labels [9] + learned graph embedding [64]', 76, 512, 510, 64, 21, { bold: true, color: C.teal });
  text(s, '→', 588, 340, 50, 85, 46, { color: C.red, align: 'center' });
  text(s, 'Expected decision output', 653, 178, 525, 40, 27, { bold: true, color: C.red });
  text(s, 'Before window t', 653, 234, 480, 33, 22, { bold: true });
  text(s, 'embedding(t−1) [64] + demand(t) [1] + queues [3]', 653, 272, 525, 60, 19);
  line(s, 653, 339, 515);
  text(s, 'DQN Q-values [6]  →  screened route ID [0–5]', 653, 357, 525, 62, 21, { bold: true });
  line(s, 653, 435, 515);
  text(s, 'After execution', 653, 455, 470, 33, 22, { bold: true });
  text(s, 'throughput  |  immediate service  |  drops\nlatency proxy  |  energy proxy  |  reward', 653, 499, 525, 76, 19);
  text(s, 'N and E vary by graph; the embedding and action dimensions stay fixed.', 76, 604, 1100, 35, 19, { color: C.muted });
  note(s, 'Source: src/phase1_pipeline.py, src/gnn_model.py, src/rl_environment.py, and config/phase4_v3_config.json. This slide contains editable native text and separators. The nine labels supervise Phase 3, but are not all DQN inputs. At decision t, only announced current demand is available from current-window data. The downstream metrics are simulator outputs.');
}

// 6. Phase 1
{
  const s = base('Phase 1: temporal graph snapshots', 6, 'DATA PREPARATION');
  text(s, 'Source', 76, 175, 250, 30, 19, { bold: true, color: C.red });
  text(s, 'Edge-IIoTset packet records', 76, 214, 525, 35, 27, { bold: true });
  text(s, 'Clean and sample 200,000 records.\nObserved traffic endpoints become nodes.\nGroup traffic into 60-second directed windows.\nFit node and edge scalers on train only.', 76, 274, 555, 155, 22);
  text(s, 'Graph tensor', 76, 483, 500, 32, 19, { bold: true, color: C.red });
  text(s, '7 node features  +  66 edge features', 76, 521, 550, 42, 27, { bold: true, color: C.teal });
  line(s, 650, 175, 0, C.gray);
  text(s, '542', 665, 190, 170, 70, 48, { bold: true, color: C.teal });
  text(s, 'TRAIN', 665, 256, 160, 28, 17, { color: C.muted });
  text(s, '113', 875, 190, 170, 70, 48, { bold: true, color: C.teal });
  text(s, 'VALIDATION', 875, 256, 180, 28, 17, { color: C.muted });
  text(s, '146', 665, 325, 170, 70, 48, { bold: true, color: C.teal });
  text(s, 'TEST', 665, 392, 160, 28, 17, { color: C.muted });
  text(s, '801 graph files total', 665, 480, 475, 40, 26, { bold: true });
  text(s, 'Output: data/processed/{split}/graph_*.pt', 665, 545, 535, 44, 20, { color: C.muted });
  note(s, 'Source: config/phase1_config.json; src/phase1_pipeline.py; graph counts cross-checked with Phase 3 evaluation metrics. Caution: two split boundaries still share nominal 60-second starts, though each file contains records from one split only.');
}

// 7. Phase 2
{
  const s = base('Phase 2: observational causal structure', 7, 'CAUSAL DISCOVERY');
  text(s, 'One row per graph', 76, 171, 450, 38, 28, { bold: true, color: C.teal });
  text(s, 'Nine retained summaries: packet volume, byte volume, packet size, interarrival time, node activity and connectivity.', 76, 224, 470, 105, 22);
  text(s, 'Training-only NOTEARS\nStability and domain refinement\nLinear structural model', 76, 371, 465, 133, 22);
  text(s, 'Output: causal_dataset.csv + refined_dag.json', 76, 545, 500, 54, 20, { bold: true });
  image(s, 'reports/phase2/final_dag.png', 585, 162, 600, 410, 'Refined observational candidate DAG generated in Phase 2');
  text(s, 'Candidate associations, not measured routing-action effects', 585, 586, 610, 39, 19, { color: C.red });
  note(s, 'Source: reports/phase2/final_dag.png and artifacts/phase2/refined_dag.json. The DAG is observational. Phase 5 safety does not use this DAG to estimate action effects.');
}

// 8. Phase 3
{
  const s = base('Phase 3: edge-aware graph encoder', 8, 'REPRESENTATION LEARNING');
  text(s, 'Directed edges carry 66 attributes', 76, 172, 480, 38, 26, { bold: true, color: C.teal });
  text(s, 'Two GATv2 blocks use four attention heads each. Separate inbound and outbound passes share information across linked endpoints.', 76, 224, 490, 128, 22);
  text(s, 'Mean + max pooling and graph size feed a 64-dimensional embedding.', 76, 390, 490, 92, 22);
  text(s, 'Training target: current nine variables + masked next-window nine variables.', 76, 504, 490, 70, 20, { bold: true });
  image(s, 'reports/phase3/training_curves.png', 600, 180, 600, 310, 'Phase 3 train and validation loss curves');
  text(s, 'Test current-state MSE 1.63 vs train-mean 4.26', 601, 525, 600, 36, 22, { bold: true, color: C.teal });
  text(s, 'Next-state forecast trails persistence on test', 601, 572, 600, 34, 18, { color: C.red });
  note(s, 'Source: src/gnn_model.py; reports/phase3/training_curves.png and evaluation_metrics.json. MSE values are standardized. Test next-state MSE 2.052 versus persistence 1.400. The GNN is not claimed to outperform simpler RL states.');
}

// 9. Phase 4 timing
{
  const s = base('Phase 4: information available before routing', 9, 'DECISION TIMING');
  text(s, 'Completed window t−1', 77, 183, 440, 37, 25, { bold: true, color: C.teal });
  text(s, 'Frozen GNN embedding', 77, 231, 440, 40, 23);
  text(s, '64 values', 77, 278, 440, 31, 19, { color: C.muted });
  line(s, 550, 182, 0);
  text(s, 'Immediately before window t', 598, 183, 575, 37, 25, { bold: true, color: C.red });
  text(s, 'Announced packet batch + current queues', 598, 231, 580, 69, 23);
  text(s, '1 demand value + 3 queue ratios', 598, 308, 580, 31, 19, { color: C.muted });
  line(s, 76, 379, 1110, C.gray, 1.4);
  text(s, '68-value DQN observation', 76, 405, 1110, 70, 38, { bold: true, color: C.ink });
  text(s, 'Graph t is completed after the action. It cannot enter the policy input for t.', 76, 499, 1110, 69, 24, { color: C.red });
  text(s, 'Legal pair requires exactly 60 seconds within one split.', 76, 590, 1100, 36, 20, { color: C.muted });
  note(s, 'Source: src/rl_environment.py and config/phase4_v3_config.json. The current packet batch is announced before routing by simulator assumption. Timing-v2 uses graph t-1 and demand t, avoiding the legacy same-window leakage. There are 537 train, 112 validation and 143 test legal decision pairs.');
}

// 10. Phase 4 agent and simulator
{
  const s = base('Phase 4: six-action DQN simulator', 10, 'REINFORCEMENT LEARNING');
  text(s, 'Three servers', 76, 176, 480, 35, 27, { bold: true, color: C.teal });
  text(s, 'A: performance   B: balanced   C: energy', 76, 221, 535, 42, 22);
  text(s, 'Actions: route to A, B or C; balanced; performance-weighted; energy-weighted.', 76, 296, 525, 105, 21);
  text(s, 'Reward favors throughput, immediate service and completion. It penalizes latency proxy, energy proxy and drops.', 76, 431, 515, 110, 21);
  image(s, 'reports/phase4_v3/reward_curve.png', 620, 196, 575, 305, 'DQN validation reward across training timesteps');
  text(s, 'Validation mean reward / action', 633, 530, 560, 35, 21, { bold: true });
  text(s, 'DQN 0.433     Random 0.408', 633, 574, 550, 43, 26, { bold: true, color: C.teal });
  note(s, 'Source: config/phase4_v3_config.json, reports/phase4_v3/reward_curve.png, reports/phase4_v3/evaluation_metrics.json. Same validation starts were used for DQN and random. Values are simulator rewards; latency and energy are synthetic proxies. DQN trained for 100,000 steps and selected checkpoint on validation.');
}

// 11. Phase 5 method
{
  const s = base('Phase 5: counterfactual action screening', 11, 'SAFETY CONTROLLER');
  text(s, '1', 80, 185, 58, 46, 33, { bold: true, color: C.red });
  text(s, 'Rank all six DQN Q-values', 146, 187, 890, 45, 27, { bold: true });
  line(s, 80, 245, 1040);
  text(s, '2', 80, 270, 58, 46, 33, { bold: true, color: C.red });
  text(s, 'Exact simulator SCM tests each route from the same demand and queues', 146, 271, 1015, 70, 25, { bold: true });
  line(s, 80, 360, 1040);
  text(s, '3', 80, 385, 58, 46, 33, { bold: true, color: C.red });
  text(s, 'Execute highest-ranked safe action, or label fallback', 146, 386, 1010, 70, 25, { bold: true });
  line(s, 80, 475, 1040);
  text(s, 'Safety limits', 80, 506, 285, 39, 22, { bold: true, color: C.teal });
  text(s, 'Latency ≤15,000 ms   Energy ≤5.28   Drops ≤1%   Immediate service ≥99%', 80, 552, 1100, 62, 22);
  note(s, 'Source: src/counterfactual_controller.py and config/phase5_v3_config.json. Exact SCM independently reproduces the simulator equations, not Phase 2 observational DAG. Q-values rank actions and are not probabilities. When no action is safe, minimum normalized violation fallback is explicitly recorded as unsafe. A learned ExtraTrees surrogate is diagnostic only.');
}

// 12. Phase 5 evidence
{
  const s = base('Phase 5: checked outcomes and edge cases', 12, 'CONTROLLER VALIDATION');
  text(s, 'Test feasible-state violations', 76, 188, 800, 40, 26, { bold: true });
  text(s, '19', 76, 251, 185, 94, 63, { bold: true, color: C.red });
  text(s, 'Unscreened DQN', 76, 341, 330, 33, 20, { color: C.muted });
  text(s, '0', 440, 251, 185, 94, 63, { bold: true, color: C.teal });
  text(s, 'Screened DQN', 440, 341, 330, 33, 20, { color: C.muted });
  line(s, 76, 407, 1080);
  text(s, '37', 76, 444, 185, 83, 54, { bold: true, color: C.red });
  text(s, 'No-safe-action fallbacks on the screened test rollout', 240, 450, 890, 70, 23);
  text(s, 'Every selected exact prediction is checked against the realized simulator step.', 76, 562, 1080, 52, 22, { color: C.teal });
  note(s, 'Source: reports/phase5_v3/violation_summary.json and evaluation_metrics.json. Figures use the saved fixed-start test evaluation (10 episodes, 484 steps); episodes can overlap, so these are simulator checks, not independent test observations. The 37 no-safe-action cases remain violations if constraints are physically infeasible. Phase 5 audit confirms all mandatory checks pass.');
}

// 13. Phase 6 evaluation scope
{
  const s = base('Phase 6: final evaluation scope', 13, 'NEXT EVALUATION');
  text(s, 'Representation comparison', 76, 176, 500, 39, 26, { bold: true, color: C.teal });
  text(s, 'Previous-window GNN embedding\nPrevious-window nine graph variables\nDemand + queues only', 76, 229, 515, 128, 22);
  text(s, 'Each DQN: with and without the exact Phase 5 screen.', 76, 374, 520, 83, 22, { bold: true });
  line(s, 620, 177, 0);
  text(s, 'Evaluation protocol', 643, 176, 510, 39, 26, { bold: true, color: C.red });
  text(s, 'Matched actions, simulator and training budget. Use validation for checkpoint selection, then assess the frozen agents on test.', 643, 229, 535, 128, 22);
  text(s, 'Stress announced demand by 1.5× and 2×.', 643, 374, 515, 81, 22, { bold: true });
  line(s, 76, 489, 1100);
  text(s, 'Report reward, immediate service, drops, latency and energy proxies, feasible-state violations, fallbacks and six-action explanations.', 76, 512, 1100, 86, 22);
  text(s, 'Interpretation remains limited to the simulator and its announced-demand assumption.', 76, 614, 1100, 38, 19, { color: C.muted });
  note(s, 'Source: config/phase6_config.json and src/phase6_pipeline.py. The repository already contains Phase 6 outputs, but this interim deck intentionally presents results only through Phase 5. Explain Phase 6 as the evaluation protocol to be covered in the final review. The full implementation compares five seeds, normal test traffic and synthetic demand surges. It does not validate real-network safety or attack detection.');
}

await fs.mkdir(build, { recursive: true });
await fs.mkdir(output, { recursive: true });
for (let i = 0; i < pres.slides.items.length; i++) {
  const slide = pres.slides.items[i];
  const png = await pres.export({ slide, format: 'png', scale: 1 });
  await fs.writeFile(path.join(build, `slide-${String(i + 1).padStart(2, '0')}.png`), new Uint8Array(await png.arrayBuffer()));
}
await (await PresentationFile.exportPptx(pres)).save(draftPath);

const { finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const result = await finalizePresentation({
  explicitTotalSlideCount: 13,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir: root,
  candidatePath: draftPath,
  finalPath,
  pythonExecutable: 'C:/Users/surya/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-heading-fit'],
  fontPolicy: { basis: 'design', families: [font] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(build, 'validation-v5.json'),
});
console.log(JSON.stringify({ finalPath, result }, null, 2));
