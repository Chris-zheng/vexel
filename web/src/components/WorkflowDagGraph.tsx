import { useMemo, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  Database,
  FileSpreadsheet,
  GripVertical,
  Trash2,
} from 'lucide-react';
import type { Lang, ModelCard, OperatorSpec, UnifiedRecipe, WorkflowStep } from '../types/vexel';
import { OPERATOR_CATALOG } from '../engine/operators';

interface WorkflowDagGraphProps {
  lang: Lang;
  recipe: UnifiedRecipe;
  models: ModelCard[];
  selectedStepId: string | null;
  onSelectStep: (stepId: string | null) => void;
  onReorderSteps: (fromIndex: number, toIndex: number) => void;
  onToggleStep: (stepId: string) => void;
  onRemoveStep: (stepId: string) => void;
  onUpdateParam: (stepId: string, key: string, value: number | boolean | string) => void;
  onUpdateModelRef: (stepId: string, modelRef: string) => void;
}

interface DagNodeLayout {
  id: string;
  kind: 'input' | 'step' | 'output';
  step?: WorkflowStep;
  stepIndex?: number;
  spec?: OperatorSpec;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface DagEdge {
  id: string;
  fromId: string;
  toId: string;
  label: string;
  channel: 'image' | 'mask' | 'objects' | 'bypass';
  isBypass?: boolean;
  arcOffsetY?: number;
}

const NODE_W = 204;
const NODE_H = 104;
const TERMINAL_W = 148;
const TERMINAL_H = 88;
const COL_GAP = 64;
const PAD_X = 28;
const BASE_Y = 76;

const CATEGORY_META: Record<
  OperatorSpec['category'],
  { accent: string; text: string; zh: string; en: string }
> = {
  preprocess: {
    accent: 'bg-cyan-500',
    text: 'text-cyan-700 dark:text-cyan-400',
    zh: '预处理',
    en: 'PREPROCESS',
  },
  segment: {
    accent: 'bg-violet-500',
    text: 'text-violet-700 dark:text-violet-400',
    zh: '分割',
    en: 'SEGMENT',
  },
  model: {
    accent: 'bg-indigo-500',
    text: 'text-indigo-700 dark:text-indigo-400',
    zh: 'AI 模型',
    en: 'MODEL',
  },
  metrology: {
    accent: 'bg-emerald-500',
    text: 'text-emerald-700 dark:text-emerald-400',
    zh: '定量计量',
    en: 'METROLOGY',
  },
  filter: {
    accent: 'bg-amber-500',
    text: 'text-amber-700 dark:text-amber-400',
    zh: '过滤质控',
    en: 'FILTER',
  },
};

export function WorkflowDagGraph({
  lang,
  recipe,
  models,
  selectedStepId,
  onSelectStep,
  onReorderSteps,
  onToggleStep,
  onRemoveStep,
  onUpdateParam,
  onUpdateModelRef,
}: WorkflowDagGraphProps) {
  const [dragFromIdx, setDragFromIdx] = useState<number | null>(null);
  const [dropSlotIdx, setDropSlotIdx] = useState<number | null>(null);

  // Compute deterministic DAG node layout and typed data-flow edges
  const { nodes, edges, canvasWidth, canvasHeight } = useMemo(() => {
    const outNodes: DagNodeLayout[] = [];
    const nodeById = new Map<string, DagNodeLayout>();

    let curX = PAD_X;
    const inputNode: DagNodeLayout = {
      id: '__input__',
      kind: 'input',
      x: curX,
      y: BASE_Y + (NODE_H - TERMINAL_H) / 2,
      width: TERMINAL_W,
      height: TERMINAL_H,
    };
    outNodes.push(inputNode);
    nodeById.set(inputNode.id, inputNode);
    curX += TERMINAL_W + COL_GAP;

    // Detect if there is an enabled model/segment branch feeding a downstream consumer that also reads image
    const hasBranchMerge = recipe.steps.some((s, idx) => {
      if (!s.enabled) return false;
      const sp = OPERATOR_CATALOG.find((o) => o.op === s.op);
      if (sp?.category !== 'model' && sp?.category !== 'segment') return false;
      return recipe.steps.slice(idx + 1).some((d) => {
        if (!d.enabled) return false;
        const dsp = OPERATOR_CATALOG.find((o) => o.op === d.op);
        return (
          dsp?.category === 'metrology' ||
          Boolean(dsp?.inputs.includes('parent_mask') && dsp?.inputs.includes('image'))
        );
      });
    });

    recipe.steps.forEach((step, idx) => {
      const spec = OPERATOR_CATALOG.find((o) => o.op === step.op);
      const isBranchNode =
        hasBranchMerge &&
        step.enabled &&
        (spec?.category === 'model' ||
          (spec?.category === 'segment' && !spec?.inputs.includes('parent_mask')));
      const node: DagNodeLayout = {
        id: step.id,
        kind: 'step',
        step,
        stepIndex: idx,
        spec,
        x: curX,
        y: isBranchNode ? BASE_Y + 46 : BASE_Y,
        width: NODE_W,
        height: NODE_H,
      };
      outNodes.push(node);
      nodeById.set(node.id, node);
      curX += NODE_W + COL_GAP;
    });

    const outputNode: DagNodeLayout = {
      id: '__output__',
      kind: 'output',
      x: curX,
      y: BASE_Y + (NODE_H - TERMINAL_H) / 2,
      width: TERMINAL_W,
      height: TERMINAL_H,
    };
    outNodes.push(outputNode);
    nodeById.set(outputNode.id, outputNode);

    const outEdges: DagEdge[] = [];
    let lastImageId = '__input__';
    let lastMaskId: string | null = null;
    let lastParentMaskId: string | null = null;
    let lastSubMaskId: string | null = null;
    let lastObjectsId: string | null = null;

    recipe.steps.forEach((step, idx) => {
      const spec = OPERATOR_CATALOG.find((o) => o.op === step.op);
      const cat = spec?.category ?? 'preprocess';
      const inputs = spec?.inputs ?? [];
      const outputs = spec?.outputs ?? [];

      if (!step.enabled) {
        // Connect faint dashed structural reference to the previous step for skipped steps
        const prevId = idx === 0 ? '__input__' : recipe.steps[idx - 1].id;
        outEdges.push({
          id: `skip_${prevId}_${step.id}`,
          fromId: prevId,
          toId: step.id,
          label: lang === 'zh' ? '跳过' : 'skip',
          channel: 'bypass',
          isBypass: true,
        });
        return;
      }

      if (cat === 'preprocess') {
        outEdges.push({
          id: `img_${lastImageId}_${step.id}`,
          fromId: lastImageId,
          toId: step.id,
          label: 'image',
          channel: 'image',
        });
        lastImageId = step.id;
      } else if (cat === 'model' || cat === 'segment') {
        const wantsParent = inputs.includes('parent_mask');
        if (wantsParent && lastParentMaskId) {
          // Constrained sub-region segmentation receives BOTH parent_mask AND image
          outEdges.push({
            id: `img_${lastImageId}_${step.id}`,
            fromId: lastImageId,
            toId: step.id,
            label: 'image',
            channel: 'image',
            arcOffsetY: -54,
          });
          outEdges.push({
            id: `pmask_${lastParentMaskId}_${step.id}`,
            fromId: lastParentMaskId,
            toId: step.id,
            label: 'parent_mask',
            channel: 'mask',
          });
        } else {
          outEdges.push({
            id: `img_${lastImageId}_${step.id}`,
            fromId: lastImageId,
            toId: step.id,
            label: 'image',
            channel: 'image',
          });
        }
        if (outputs.includes('parent_mask') && !wantsParent) {
          lastParentMaskId = step.id;
        }
        if (outputs.includes('subregion_mask')) {
          lastSubMaskId = step.id;
        }
        lastMaskId = step.id;
      } else if (cat === 'metrology') {
        const isHierarchyMetrology = inputs.includes('subregion_mask') && inputs.includes('parent_mask');
        if (isHierarchyMetrology && (lastSubMaskId || lastParentMaskId)) {
          if (lastSubMaskId) {
            outEdges.push({
              id: `sub_${lastSubMaskId}_${step.id}`,
              fromId: lastSubMaskId,
              toId: step.id,
              label: 'subregion_mask',
              channel: 'mask',
            });
          }
          if (lastParentMaskId && lastParentMaskId !== lastSubMaskId) {
            outEdges.push({
              id: `par_${lastParentMaskId}_${step.id}`,
              fromId: lastParentMaskId,
              toId: step.id,
              label: 'parent_mask',
              channel: 'mask',
              arcOffsetY: 56,
            });
          }
        } else {
          // Multi-parent DAG merge when an enabled model/segment branch feeds alongside preprocessed image
          const hasMaskParent = lastMaskId !== null;
          outEdges.push({
            id: `img_${lastImageId}_${step.id}`,
            fromId: lastImageId,
            toId: step.id,
            label: 'image',
            channel: 'image',
            arcOffsetY: hasMaskParent ? -54 : 0,
          });
          if (lastMaskId) {
            outEdges.push({
              id: `mask_${lastMaskId}_${step.id}`,
              fromId: lastMaskId,
              toId: step.id,
              label: 'prob_map',
              channel: 'mask',
            });
          }
        }
        lastObjectsId = step.id;
      } else if (cat === 'filter') {
        const srcId = lastObjectsId ?? lastMaskId ?? lastImageId;
        outEdges.push({
          id: `obj_${srcId}_${step.id}`,
          fromId: srcId,
          toId: step.id,
          label: lastObjectsId ? 'objects' : 'image',
          channel: lastObjectsId ? 'objects' : 'image',
        });
        lastObjectsId = step.id;
      }
    });

    const finalProducer = lastObjectsId ?? lastMaskId ?? lastImageId;
    outEdges.push({
      id: `out_${finalProducer}___output__`,
      fromId: finalProducer,
      toId: '__output__',
      label: lastObjectsId ? 'table · overlay' : lastMaskId ? 'mask' : 'image',
      channel: lastObjectsId ? 'objects' : lastMaskId ? 'mask' : 'image',
    });

    const totalW = curX + TERMINAL_W + PAD_X;
    const totalH = hasBranchMerge ? BASE_Y + NODE_H + 92 : BASE_Y + NODE_H + 54;

    return {
      nodes: outNodes,
      edges: outEdges,
      canvasWidth: totalW,
      canvasHeight: totalH,
    };
  }, [recipe.steps, lang]);

  const nodeMap = useMemo(() => {
    const m = new Map<string, DagNodeLayout>();
    for (const n of nodes) m.set(n.id, n);
    return m;
  }, [nodes]);

  const activeSelectedStep =
    recipe.steps.find((s) => s.id === selectedStepId) ?? recipe.steps[0] ?? null;
  const activeSelectedIdx = activeSelectedStep
    ? recipe.steps.findIndex((s) => s.id === activeSelectedStep.id)
    : -1;
  const activeSelectedSpec = activeSelectedStep
    ? OPERATOR_CATALOG.find((o) => o.op === activeSelectedStep.op)
    : undefined;

  // Commit drag-and-drop reorder from dragFromIdx to insertion slot slotIdx (0..steps.length)
  const commitDropAtSlot = (slotIdx: number) => {
    if (dragFromIdx === null) return;
    let targetIndex = slotIdx;
    if (dragFromIdx < slotIdx) targetIndex = slotIdx - 1;
    if (targetIndex !== dragFromIdx && targetIndex >= 0 && targetIndex < recipe.steps.length) {
      onReorderSteps(dragFromIdx, targetIndex);
    }
    setDragFromIdx(null);
    setDropSlotIdx(null);
  };

  const edgeStroke = (ch: DagEdge['channel']) => {
    if (ch === 'mask') return '#8B5CF6';
    if (ch === 'objects') return '#10B981';
    if (ch === 'bypass') return '#94A3B8';
    return '#06B6D4';
  };

  return (
    <div className="space-y-4">
      {/* DAG Viewport Card */}
      <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0B0F1A] overflow-hidden">
        {/* Compact Graph Telemetry Header */}
        <div className="px-4 py-2.5 border-b border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-[#0E1320] flex flex-wrap items-center justify-between gap-2 text-[11.5px]">
          <div className="flex items-center gap-2 text-slate-600 dark:text-slate-300">
            <span className="font-semibold text-slate-900 dark:text-slate-100">
              {lang === 'zh' ? '有向无环图 (DAG) 算子流' : 'Directed Acyclic Graph (DAG)'}
            </span>
            <span className="text-slate-400">·</span>
            <span className="font-mono tabular-nums text-slate-500">
              {recipe.steps.filter((s) => s.enabled).length}/{recipe.steps.length}{' '}
              {lang === 'zh' ? '启用节点' : 'active'}
            </span>
            <span className="text-slate-400 hidden sm:inline">·</span>
            <span className="text-slate-500 hidden sm:inline">
              {lang === 'zh'
                ? '拖拽节点卡片可重新排序，点击节点编辑参数'
                : 'Drag nodes to reorder pipeline · Click a node to edit parameters'}
            </span>
          </div>
          <div className="flex items-center gap-3 font-mono text-[10.5px] text-slate-500">
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-0.5 bg-cyan-500" />
              image
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-0.5 bg-violet-500" />
              prob_map
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-0.5 bg-emerald-500" />
              objects
            </span>
          </div>
        </div>

        {/* Scrollable Interactive DAG Canvas */}
        <div
          className="relative overflow-x-auto overflow-y-hidden select-none"
          style={{
            backgroundImage:
              'radial-gradient(circle, rgba(148,163,184,0.22) 1px, transparent 1px)',
            backgroundSize: '18px 18px',
          }}
          onDragOver={(e) => {
            if (dragFromIdx !== null) e.preventDefault();
          }}
        >
          <div
            className="relative"
            style={{ width: `${canvasWidth}px`, height: `${canvasHeight}px`, minWidth: '100%' }}
          >
            {/* SVG Directed Edges Layer */}
            <svg
              width={canvasWidth}
              height={canvasHeight}
              className="absolute inset-0 pointer-events-none"
            >
              <defs>
                <marker
                  id="dag-arrow-image"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#06B6D4" />
                </marker>
                <marker
                  id="dag-arrow-mask"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#8B5CF6" />
                </marker>
                <marker
                  id="dag-arrow-objects"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#10B981" />
                </marker>
                <marker
                  id="dag-arrow-bypass"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="5.5"
                  markerHeight="5.5"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 2 L 8 5 L 0 8 z" fill="#94A3B8" opacity="0.6" />
                </marker>
              </defs>

              {edges.map((edge) => {
                const from = nodeMap.get(edge.fromId);
                const to = nodeMap.get(edge.toId);
                if (!from || !to) return null;

                const x1 = from.x + from.width;
                const y1 = from.y + from.height / 2;
                const x2 = to.x;
                const y2 =
                  edge.channel === 'mask'
                    ? to.y + to.height * 0.72
                    : edge.arcOffsetY
                    ? to.y + to.height * 0.28
                    : to.y + to.height / 2;

                const dx = Math.max(36, (x2 - x1) * 0.48);
                let pathD: string;
                let labelX: number;
                let labelY: number;

                if (edge.arcOffsetY) {
                  // Elevated skip arc over an intermediate branch node (true multi-parent DAG edge)
                  const topY = Math.min(from.y, to.y) + edge.arcOffsetY;
                  pathD = `M ${x1} ${y1} C ${x1 + 42} ${topY}, ${x2 - 42} ${topY}, ${x2} ${y2}`;
                  labelX = (x1 + x2) * 0.5;
                  labelY = topY + 12;
                } else {
                  pathD = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
                  labelX = (x1 + x2) * 0.5;
                  labelY = (y1 + y2) * 0.5 - 8;
                }

                const stroke = edgeStroke(edge.channel);
                return (
                  <g key={edge.id}>
                    <path
                      d={pathD}
                      fill="none"
                      stroke={stroke}
                      strokeWidth={edge.isBypass ? 1.4 : 2}
                      strokeDasharray={edge.isBypass ? '5 4' : undefined}
                      opacity={edge.isBypass ? 0.55 : 0.9}
                      markerEnd={`url(#dag-arrow-${edge.channel})`}
                    />
                    <text
                      x={labelX}
                      y={labelY}
                      textAnchor="middle"
                      className="fill-slate-500 dark:fill-slate-400 font-mono text-[9.5px]"
                    >
                      {edge.label}
                    </text>
                  </g>
                );
              })}
            </svg>

            {/* Explicit Drag-and-Drop Insertion Slot Rails between step nodes */}
            {dragFromIdx !== null &&
              Array.from({ length: recipe.steps.length + 1 }, (_, slotIdx) => {
                const firstStepX = PAD_X + TERMINAL_W + COL_GAP;
                const slotCenterX =
                  slotIdx === 0
                    ? firstStepX - COL_GAP * 0.5
                    : firstStepX + slotIdx * (NODE_W + COL_GAP) - COL_GAP * 0.5;
                const isActiveSlot = dropSlotIdx === slotIdx;
                return (
                  <div
                    key={`slot_${slotIdx}`}
                    style={{
                      left: `${slotCenterX - 24}px`,
                      top: `${BASE_Y - 18}px`,
                      width: '48px',
                      height: `${NODE_H + 64}px`,
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      e.dataTransfer.dropEffect = 'move';
                      if (dropSlotIdx !== slotIdx) setDropSlotIdx(slotIdx);
                    }}
                    onDrop={(e) => {
                      e.preventDefault();
                      commitDropAtSlot(slotIdx);
                    }}
                    className="absolute z-20 flex flex-col items-center justify-center"
                  >
                    <div
                      className={`w-1 rounded-full transition-all duration-150 ${
                        isActiveSlot
                          ? 'h-full bg-blue-500 dark:bg-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.7)]'
                          : 'h-3/4 bg-slate-300/60 dark:bg-slate-700/60'
                      }`}
                    />
                    {isActiveSlot && (
                      <span className="absolute -top-5 px-1.5 py-0.5 rounded bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 font-mono text-[9.5px] font-semibold whitespace-nowrap">
                        #{String(slotIdx + 1).padStart(2, '0')}
                      </span>
                    )}
                  </div>
                );
              })}

            {/* DAG Nodes Layer */}
            {nodes.map((node) => {
              if (node.kind === 'input') {
                return (
                  <div
                    key={node.id}
                    style={{
                      left: `${node.x}px`,
                      top: `${node.y}px`,
                      width: `${node.width}px`,
                      height: `${node.height}px`,
                    }}
                    className="absolute rounded-md border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-[#111726] p-3 flex flex-col justify-between"
                  >
                    <div className="flex items-center justify-between text-[10px] font-mono text-slate-500">
                      <span>SOURCE</span>
                      <Database className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" />
                    </div>
                    <div>
                      <div className="text-[12px] font-semibold text-slate-900 dark:text-slate-100">
                        {lang === 'zh' ? '输入图像 / ROI' : 'Input Specimen'}
                      </div>
                      <div className="font-mono text-[10px] text-slate-500">
                        {lang === 'zh' ? '标定后有效视场' : 'calibrated FOV'}
                      </div>
                    </div>
                    {/* Right output port */}
                    <span className="absolute -right-1.5 top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-cyan-500 border-2 border-white dark:border-[#0B0F1A]" />
                  </div>
                );
              }

              if (node.kind === 'output') {
                return (
                  <div
                    key={node.id}
                    style={{
                      left: `${node.x}px`,
                      top: `${node.y}px`,
                      width: `${node.width}px`,
                      height: `${node.height}px`,
                    }}
                    className="absolute rounded-md border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-[#111726] p-3 flex flex-col justify-between"
                  >
                    <div className="flex items-center justify-between text-[10px] font-mono text-slate-500">
                      <span>OUTPUT</span>
                      <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                    </div>
                    <div>
                      <div className="text-[12px] font-semibold text-slate-900 dark:text-slate-100">
                        {lang === 'zh' ? '计量结果与图层' : 'Metrology Output'}
                      </div>
                      <div className="font-mono text-[10px] text-slate-500">
                        table · overlay · QC
                      </div>
                    </div>
                    {/* Left input port */}
                    <span className="absolute -left-1.5 top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-emerald-500 border-2 border-white dark:border-[#0B0F1A]" />
                  </div>
                );
              }

              const step = node.step!;
              const idx = node.stepIndex!;
              const spec = node.spec;
              const cat = spec?.category ?? 'preprocess';
              const meta = CATEGORY_META[cat];
              const isSelected = activeSelectedStep?.id === step.id;
              const isDraggingThis = dragFromIdx === idx;
              const modifiedParams = spec
                ? spec.params.filter((p) => (step.params[p.key] ?? p.defaultValue) !== p.defaultValue).length
                : 0;

              return (
                <div
                  key={node.id}
                  draggable
                  onDragStart={(e) => {
                    setDragFromIdx(idx);
                    e.dataTransfer.effectAllowed = 'move';
                    e.dataTransfer.setData('text/plain', String(idx));
                  }}
                  onDragOver={(e) => {
                    if (dragFromIdx === null || dragFromIdx === idx) return;
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                    const rect = e.currentTarget.getBoundingClientRect();
                    const insertBefore = e.clientX < rect.left + rect.width / 2;
                    const nextSlot = insertBefore ? idx : idx + 1;
                    if (dropSlotIdx !== nextSlot) setDropSlotIdx(nextSlot);
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    if (dragFromIdx === null) return;
                    const rect = e.currentTarget.getBoundingClientRect();
                    const insertBefore = e.clientX < rect.left + rect.width / 2;
                    commitDropAtSlot(insertBefore ? idx : idx + 1);
                  }}
                  onDragEnd={() => {
                    setDragFromIdx(null);
                    setDropSlotIdx(null);
                  }}
                  onClick={() => onSelectStep(step.id)}
                  style={{
                    left: `${node.x}px`,
                    top: `${node.y}px`,
                    width: `${node.width}px`,
                    height: `${node.height}px`,
                  }}
                  className={`absolute rounded-md border transition-all duration-150 cursor-grab active:cursor-grabbing flex flex-col justify-between overflow-hidden ${
                    isDraggingThis
                      ? 'opacity-40 scale-[0.98] border-blue-500 dark:border-cyan-400'
                      : !step.enabled
                      ? 'border-dashed border-slate-300 dark:border-slate-700 bg-slate-100/80 dark:bg-slate-900/40 opacity-65'
                      : isSelected
                      ? 'border-blue-600 dark:border-cyan-400 bg-white dark:bg-[#111829] ring-1 ring-blue-500/35 dark:ring-cyan-400/35'
                      : 'border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0F1422] hover:border-slate-400 dark:hover:border-slate-600'
                  }`}
                >
                  {/* Crisp 2px top category rule */}
                  <div className={`h-0.5 w-full ${step.enabled ? meta.accent : 'bg-slate-400/50'}`} />

                  <div className="px-2.5 pt-2 pb-2 flex-1 flex flex-col justify-between min-h-0">
                    {/* Top row: Drag handle + index + category + quick enable/delete */}
                    <div className="flex items-center justify-between gap-1">
                      <div className="flex items-center gap-1 min-w-0">
                        <GripVertical className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                        <span className="font-mono text-[10.5px] font-semibold tabular-nums text-slate-500">
                          {String(idx + 1).padStart(2, '0')}
                        </span>
                        <span className="text-slate-300 dark:text-slate-700">·</span>
                        <span className={`font-mono text-[10px] font-semibold truncate ${meta.text}`}>
                          {lang === 'zh' ? meta.zh : meta.en}
                        </span>
                      </div>
                      <div
                        className="flex items-center gap-1 shrink-0"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <input
                          type="checkbox"
                          checked={step.enabled}
                          onChange={() => onToggleStep(step.id)}
                          title={lang === 'zh' ? '启用/跳过节点' : 'Enable/skip node'}
                          className="accent-cyan-500 cursor-pointer"
                        />
                        <button
                          type="button"
                          onClick={() => onRemoveStep(step.id)}
                          title={lang === 'zh' ? '删除节点' : 'Delete node'}
                          className="p-0.5 text-slate-400 hover:text-rose-500 cursor-pointer"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>

                    {/* Middle: Title & operator key */}
                    <div className="min-w-0">
                      <div
                        className={`text-[12px] font-semibold truncate ${
                          step.enabled
                            ? 'text-slate-900 dark:text-slate-100'
                            : 'text-slate-500 line-through'
                        }`}
                      >
                        {spec ? (lang === 'zh' ? spec.labelZh : spec.labelEn) : step.op}
                      </div>
                      <div className="font-mono text-[10px] text-slate-500 truncate mt-0.5">
                        {step.op}
                      </div>
                    </div>

                    {/* Footer: I/O contract & modified count */}
                    <div className="flex items-center justify-between gap-1 text-[10px] font-mono text-slate-500 pt-1 border-t border-slate-100 dark:border-slate-800/80">
                      <span className="truncate">
                        {spec ? `${spec.inputs.join(',')} → ${spec.outputs[0]}` : 'io'}
                      </span>
                      {modifiedParams > 0 && (
                        <span className="text-amber-600 dark:text-amber-400 shrink-0 tabular-nums">
                          {modifiedParams} {lang === 'zh' ? '项改' : 'mod'}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Input & Output Port Connectors */}
                  <span className="absolute -left-1.5 top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-slate-300 dark:bg-slate-600 border-2 border-white dark:border-[#0B0F1A]" />
                  <span className="absolute -right-1.5 top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-cyan-500 border-2 border-white dark:border-[#0B0F1A]" />
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Selected Node Parameter Inspector (docked directly below DAG) */}
      {activeSelectedStep && (
        <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E1320]">
          <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2.5 min-w-0">
              <span className="font-mono text-[12px] font-semibold tabular-nums text-blue-600 dark:text-cyan-400">
                #{String(activeSelectedIdx + 1).padStart(2, '0')}
              </span>
              <div className="min-w-0">
                <div className="text-[13.5px] font-semibold text-slate-900 dark:text-slate-100 truncate">
                  {activeSelectedSpec
                    ? lang === 'zh'
                      ? activeSelectedSpec.labelZh
                      : activeSelectedSpec.labelEn
                    : activeSelectedStep.op}
                </div>
                <div className="font-mono text-[11px] text-slate-500">
                  {activeSelectedStep.op}
                  {activeSelectedSpec
                    ? ` · ${activeSelectedSpec.inputs.join(', ')} → ${activeSelectedSpec.outputs.join(', ')}`
                    : ''}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-1.5 shrink-0">
              <label className="flex items-center gap-1.5 text-[11.5px] text-slate-600 dark:text-slate-400 mr-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={activeSelectedStep.enabled}
                  onChange={() => onToggleStep(activeSelectedStep.id)}
                  className="accent-cyan-500"
                />
                <span>
                  {activeSelectedStep.enabled
                    ? lang === 'zh'
                      ? '启用'
                      : 'Active'
                    : lang === 'zh'
                    ? '跳过'
                    : 'Skipped'}
                </span>
              </label>
              <button
                type="button"
                disabled={activeSelectedIdx <= 0}
                onClick={() => onReorderSteps(activeSelectedIdx, activeSelectedIdx - 1)}
                title={lang === 'zh' ? '前移一位' : 'Move earlier'}
                className="h-7 px-2 rounded border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-slate-100 disabled:opacity-30 flex items-center gap-1 text-[11px] cursor-pointer"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                <span>{lang === 'zh' ? '前移' : 'Earlier'}</span>
              </button>
              <button
                type="button"
                disabled={activeSelectedIdx < 0 || activeSelectedIdx >= recipe.steps.length - 1}
                onClick={() => onReorderSteps(activeSelectedIdx, activeSelectedIdx + 1)}
                title={lang === 'zh' ? '后移一位' : 'Move later'}
                className="h-7 px-2 rounded border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-slate-100 disabled:opacity-30 flex items-center gap-1 text-[11px] cursor-pointer"
              >
                <span>{lang === 'zh' ? '后移' : 'Later'}</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => onRemoveStep(activeSelectedStep.id)}
                title={lang === 'zh' ? '移除节点' : 'Remove node'}
                className="w-7 h-7 rounded border border-slate-200 dark:border-slate-700 flex items-center justify-center text-slate-500 hover:text-rose-500 cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="px-4 py-3 grid grid-cols-1 md:grid-cols-2 gap-3.5 text-[12px]">
            {activeSelectedStep.op === 'model.segment' && (
              <label className="md:col-span-2 space-y-1">
                <span className="text-[11px] text-slate-500 dark:text-slate-400 block">
                  {lang === 'zh' ? 'Trained Model 权重绑定' : 'Trained Model Binding'}
                </span>
                <select
                  value={activeSelectedStep.modelRef ?? models[0]?.ref ?? ''}
                  onChange={(e) => onUpdateModelRef(activeSelectedStep.id, e.target.value)}
                  className="w-full h-8 px-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 font-mono text-[12px] text-slate-900 dark:text-slate-100"
                >
                  {models.map((m) => (
                    <option key={m.ref} value={m.ref}>
                      {m.ref} · {lang === 'zh' ? m.nameZh : m.nameEn}
                    </option>
                  ))}
                </select>
              </label>
            )}

            {activeSelectedSpec?.params.map((p) => {
              const val = activeSelectedStep.params[p.key] ?? p.defaultValue;
              return (
                <div key={p.key} className="space-y-1">
                  <div className="flex justify-between text-[11.5px]">
                    <span className="text-slate-700 dark:text-slate-300">
                      {lang === 'zh' ? p.labelZh : p.labelEn}
                    </span>
                    {p.type === 'number' && (
                      <span className="font-mono tabular-nums text-slate-500">
                        {String(val)} {p.unit ?? ''}
                      </span>
                    )}
                  </div>
                  {p.type === 'number' && (
                    <input
                      type="range"
                      min={p.min ?? 0}
                      max={p.max ?? 100}
                      step={p.step ?? 1}
                      value={Number(val)}
                      onChange={(e) =>
                        onUpdateParam(activeSelectedStep.id, p.key, Number(e.target.value))
                      }
                      className="w-full accent-blue-600 dark:accent-cyan-400"
                    />
                  )}
                  {p.type === 'boolean' && (
                    <label className="inline-flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={Boolean(val)}
                        onChange={(e) =>
                          onUpdateParam(activeSelectedStep.id, p.key, e.target.checked)
                        }
                        className="accent-cyan-500"
                      />
                      <span className="text-slate-700 dark:text-slate-300">
                        {Boolean(val)
                          ? lang === 'zh'
                            ? '开启'
                            : 'Enabled'
                          : lang === 'zh'
                          ? '关闭'
                          : 'Disabled'}
                      </span>
                    </label>
                  )}
                  {p.type === 'choice' && p.choices && (
                    <select
                      value={String(val)}
                      onChange={(e) => onUpdateParam(activeSelectedStep.id, p.key, e.target.value)}
                      className="w-full h-8 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12px] text-slate-900 dark:text-slate-100"
                    >
                      {p.choices.map((ch) => (
                        <option key={ch.value} value={ch.value}>
                          {lang === 'zh' ? ch.labelZh : ch.labelEn}
                        </option>
                      ))}
                    </select>
                  )}
                  {(p.helpZh || p.helpEn) && (
                    <p className="text-[10.5px] leading-snug text-slate-500 dark:text-slate-400">
                      {lang === 'zh' ? p.helpZh ?? p.helpEn : p.helpEn ?? p.helpZh}
                    </p>
                  )}
                </div>
              );
            })}

            {(!activeSelectedSpec || activeSelectedSpec.params.length === 0) &&
              activeSelectedStep.op !== 'model.segment' && (
                <div className="md:col-span-2 text-[12px] text-slate-500">
                  {lang === 'zh' ? '该算子无需额外配置参数。' : 'No configurable parameters for this node.'}
                </div>
              )}
          </div>
        </div>
      )}
    </div>
  );
}
