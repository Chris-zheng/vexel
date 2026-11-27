import { useRef, useState } from 'react';
import {
  ArrowDown,
  ArrowUp,
  BookmarkPlus,
  Check,
  Download,
  GitBranch,
  List,
  Play,
  Plus,
  Sparkles,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import type { Lang, ModelCard, OperatorSpec, UnifiedRecipe, WorkflowStep } from '../types/vexel';
import { BUILTIN_RECIPES, OPERATOR_CATALOG } from '../engine/operators';
import { t } from '../i18n/strings';
import { WorkflowDagGraph } from './WorkflowDagGraph';
import { useClickOutside } from '../hooks/useClickOutside';

interface WorkflowStudioProps {
  lang: Lang;
  recipe: UnifiedRecipe;
  models: ModelCard[];
  savedWorkflows: UnifiedRecipe[];
  onChangeRecipe: (next: UnifiedRecipe) => void;
  onSaveCustomWorkflow: (customName: string, recipeToSave: UnifiedRecipe) => void;
  onDeleteCustomWorkflow: (workflowId: string) => void;
  onApplyAndRun: (next: UnifiedRecipe) => void;
  onExportVexflow: () => void;
  onAiEditWorkflow: (goal: string) => Promise<void>;
  aiLoading: boolean;
}

export function WorkflowStudio({
  lang,
  recipe,
  models,
  savedWorkflows,
  onChangeRecipe,
  onSaveCustomWorkflow,
  onDeleteCustomWorkflow,
  onApplyAndRun,
  onExportVexflow,
  onAiEditWorkflow,
  aiLoading,
}: WorkflowStudioProps) {
  const [aiPrompt, setAiPrompt] = useState('');
  const [importError, setImportError] = useState<string | null>(null);
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveName, setSaveName] = useState('');
  const saveFormRef = useRef<HTMLFormElement>(null);
  useClickOutside(saveOpen, [saveFormRef], () => setSaveOpen(false));
  const [viewMode, setViewMode] = useState<'dag' | 'list'>('dag');
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(
    () => recipe.steps[recipe.steps.length - 1]?.id ?? null
  );
  const [selectedCategory, setSelectedCategory] = useState<OperatorSpec['category']>('preprocess');
  const [selectedOpId, setSelectedOpId] = useState<string>(
    OPERATOR_CATALOG.find((o) => o.category === 'preprocess')?.op ?? OPERATOR_CATALOG[0]?.op ?? ''
  );
  const fileInputRef = useRef<HTMLInputElement>(null);

  const updateSteps = (steps: WorkflowStep[]) => {
    onChangeRecipe({ ...recipe, steps });
  };

  const reorderSteps = (fromIndex: number, toIndex: number) => {
    if (
      fromIndex === toIndex ||
      fromIndex < 0 ||
      toIndex < 0 ||
      fromIndex >= recipe.steps.length ||
      toIndex >= recipe.steps.length
    ) {
      return;
    }
    const next = [...recipe.steps];
    const [moved] = next.splice(fromIndex, 1);
    next.splice(toIndex, 0, moved);
    updateSteps(next);
  };

  const moveStep = (index: number, delta: -1 | 1) => {
    reorderSteps(index, index + delta);
  };

  const removeStep = (id: string) => {
    const remaining = recipe.steps.filter((s) => s.id !== id);
    updateSteps(remaining);
    if (selectedNodeId === id) {
      setSelectedNodeId(remaining[0]?.id ?? null);
    }
  };

  const toggleStep = (id: string) => {
    updateSteps(recipe.steps.map((s) => (s.id === id ? { ...s, enabled: !s.enabled } : s)));
  };

  const updateParam = (stepId: string, key: string, value: number | boolean | string) => {
    updateSteps(
      recipe.steps.map((s) =>
        s.id === stepId ? { ...s, params: { ...s.params, [key]: value } } : s
      )
    );
  };

  const updateModelRef = (stepId: string, modelRef: string) => {
    updateSteps(recipe.steps.map((s) => (s.id === stepId ? { ...s, modelRef } : s)));
  };

  const addOperator = (spec: OperatorSpec) => {
    const defaults: Record<string, number | boolean | string> = {};
    for (const p of spec.params) defaults[p.key] = p.defaultValue;
    const newStep: WorkflowStep = {
      id: `${spec.op.replace('.', '_')}_${Date.now().toString(36).slice(-4)}`,
      op: spec.op,
      enabled: true,
      params: defaults,
      modelRef: spec.op === 'model.segment' ? models[0]?.ref : undefined,
    };

    // Place preprocessing/segmentation/model before metrology & filtering steps by default
    const next = [...recipe.steps];
    if (spec.category === 'preprocess' || spec.category === 'segment' || spec.category === 'model') {
      const firstDownstream = next.findIndex((s) => {
        const os = OPERATOR_CATALOG.find((o) => o.op === s.op);
        return os?.category === 'metrology' || os?.category === 'filter';
      });
      if (firstDownstream >= 0) next.splice(firstDownstream, 0, newStep);
      else next.push(newStep);
    } else {
      next.push(newStep);
    }
    updateSteps(next);
    setSelectedNodeId(newStep.id);
  };

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImportError(null);
    try {
      const txt = await file.text();
      const parsed = JSON.parse(txt);
      const imported: UnifiedRecipe = parsed.recipe ?? parsed;
      if (!imported || !Array.isArray(imported.steps) || !imported.id) {
        throw new Error('Invalid .vexflow schema');
      }
      onChangeRecipe(imported);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : 'Failed to parse .vexflow JSON');
    } finally {
      e.target.value = '';
    }
  };

  const categories: Array<{ id: OperatorSpec['category']; zh: string; en: string }> = [
    { id: 'preprocess', zh: '预处理算子 (Preprocess)', en: 'Preprocessing' },
    { id: 'segment', zh: '图像分割算子 (Segmentation)', en: 'Segmentation' },
    { id: 'model', zh: 'Trained Model 智能节点', en: 'Trained Models' },
    { id: 'metrology', zh: '定量计量分析 (Metrology)', en: 'Quantitative Metrology' },
    { id: 'filter', zh: '目标过滤与质控 (Filter & QC)', en: 'Filtering & QC' },
  ];

  const categoryOps = OPERATOR_CATALOG.filter((o) => o.category === selectedCategory);
  const selectedOperator =
    categoryOps.find((o) => o.op === selectedOpId) ?? categoryOps[0] ?? null;

  const handleChangeCategory = (nextCat: OperatorSpec['category']) => {
    setSelectedCategory(nextCat);
    const firstOp = OPERATOR_CATALOG.find((o) => o.category === nextCat);
    if (firstOp) setSelectedOpId(firstOp.op);
  };

  const handleSelectPresetId = (presetId: string) => {
    const found =
      savedWorkflows.find((w) => w.id === presetId) ??
      BUILTIN_RECIPES.find((r) => r.id === presetId);
    if (found) {
      onChangeRecipe(JSON.parse(JSON.stringify(found)));
    }
  };

  const isCustomActive = savedWorkflows.some((w) => w.id === recipe.id);
  const bioRecipes = BUILTIN_RECIPES.filter((r) =>
    ['droplet_encapsulation', 'droplet_fluorescence', 'cell_segmentation'].includes(r.id)
  );
  const semRecipes = BUILTIN_RECIPES.filter((r) =>
    ['sem_lines', 'sem_holes', 'sem_particles', 'sem_fibres'].includes(r.id)
  );
  const generalRecipes = BUILTIN_RECIPES.filter(
    (r) => !bioRecipes.includes(r) && !semRecipes.includes(r)
  );

  return (
    <div className="flex-1 min-h-0 overflow-hidden grid grid-cols-1 lg:grid-cols-12 bg-slate-50 dark:bg-[#0B0E17]">
      {/* Left Column: Categorized Dropdowns for Workflow Templates & Processing Operators */}
      <div className="lg:col-span-4 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E1320] overflow-y-auto p-5 space-y-6">
        {/* 1. Categorized Preset & Saved Workflow Dropdown */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between gap-2">
            <h2 className="text-[13.5px] font-semibold text-slate-900 dark:text-slate-100">
              {t('wf_preset_recipes', lang)}
            </h2>
            <span className="font-mono text-[10.5px] text-slate-500">
              {BUILTIN_RECIPES.length + savedWorkflows.length} {lang === 'zh' ? '套' : 'total'}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <select
              aria-label={t('wf_preset_recipes', lang)}
              value={recipe.id}
              onChange={(e) => handleSelectPresetId(e.target.value)}
              className="flex-1 min-w-0 h-9 px-2.5 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12.5px] font-medium text-slate-900 dark:text-slate-100 cursor-pointer focus:outline-none focus:border-blue-500 dark:focus:border-cyan-500"
            >
              {savedWorkflows.length > 0 && (
                <optgroup label={lang === 'zh' ? '已保存的自定义工作流' : 'Saved Custom Workflows'}>
                  {savedWorkflows.map((custom) => (
                    <option key={custom.id} value={custom.id}>
                      ★ {custom.name} ({custom.steps.length} {lang === 'zh' ? '步' : 'steps'})
                    </option>
                  ))}
                </optgroup>
              )}
              <optgroup label={lang === 'zh' ? '微流控与生物细胞' : 'Microfluidics & Biology'}>
                {bioRecipes.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {lang === 'zh' ? preset.nameZh : preset.nameEn}
                  </option>
                ))}
              </optgroup>
              <optgroup label={lang === 'zh' ? 'SEM 半导体与材料计量' : 'SEM & Materials Metrology'}>
                {semRecipes.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {lang === 'zh' ? preset.nameZh : preset.nameEn}
                  </option>
                ))}
              </optgroup>
              <optgroup label={lang === 'zh' ? '工业检测与通用分割' : 'Industrial & General Segmentation'}>
                {generalRecipes.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {lang === 'zh' ? preset.nameZh : preset.nameEn}
                  </option>
                ))}
              </optgroup>
            </select>

            {isCustomActive && (
              <button
                type="button"
                onClick={() => onDeleteCustomWorkflow(recipe.id)}
                title={lang === 'zh' ? '删除此自定义工作流' : 'Delete custom workflow'}
                className="h-9 w-9 rounded-md border border-slate-300 dark:border-slate-700 text-slate-400 hover:text-rose-500 hover:border-rose-500/50 flex items-center justify-center shrink-0 cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          <div className="p-3 rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-900/50 space-y-1">
            <div className="flex items-center justify-between gap-2 text-[11px] font-mono text-slate-500">
              <span className="truncate">{recipe.id}</span>
              <span className="shrink-0 uppercase">
                {recipe.steps.length} {lang === 'zh' ? '步算子' : 'steps'}
              </span>
            </div>
            <p className="text-[11.5px] leading-relaxed text-slate-600 dark:text-slate-400">
              {lang === 'zh' ? recipe.descriptionZh : recipe.descriptionEn}
            </p>
          </div>
        </div>

        {/* 2. Categorized Processing Operator Dropdown + Operator Picker */}
        <div className="pt-4 border-t border-slate-200 dark:border-slate-800 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <h2 className="text-[13.5px] font-semibold text-slate-900 dark:text-slate-100">
              {t('wf_add_step', lang)}
            </h2>
            <span className="font-mono text-[10.5px] text-slate-500">
              {OPERATOR_CATALOG.length} {lang === 'zh' ? '个算子' : 'ops'}
            </span>
          </div>

          {/* Processing Category Dropdown */}
          <label className="block space-y-1">
            <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400 block">
              {lang === 'zh' ? '1. 选择图像处理类别' : '1. Processing Category'}
            </span>
            <select
              aria-label={lang === 'zh' ? '选择图像处理类别' : 'Select processing category'}
              value={selectedCategory}
              onChange={(e) => handleChangeCategory(e.target.value as OperatorSpec['category'])}
              className="w-full h-9 px-2.5 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12.5px] font-medium text-slate-900 dark:text-slate-100 cursor-pointer focus:outline-none focus:border-blue-500 dark:focus:border-cyan-500"
            >
              {categories.map((cat) => {
                const count = OPERATOR_CATALOG.filter((o) => o.category === cat.id).length;
                if (count === 0) return null;
                return (
                  <option key={cat.id} value={cat.id}>
                    {lang === 'zh' ? cat.zh : cat.en} ({count})
                  </option>
                );
              })}
            </select>
          </label>

          {/* Operator Dropdown filtered by chosen category + Add CTA */}
          <label className="block space-y-1">
            <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400 block">
              {lang === 'zh' ? '2. 选择具体处理算子' : '2. Select Operator'}
            </span>
            <div className="flex items-center gap-1.5">
              <select
                aria-label={lang === 'zh' ? '选择具体处理算子' : 'Select processing operator'}
                value={selectedOperator?.op ?? ''}
                onChange={(e) => setSelectedOpId(e.target.value)}
                className="flex-1 min-w-0 h-9 px-2.5 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12.5px] text-slate-900 dark:text-slate-100 cursor-pointer focus:outline-none focus:border-blue-500 dark:focus:border-cyan-500"
              >
                {categoryOps.map((op) => (
                  <option key={op.op} value={op.op}>
                    {lang === 'zh' ? op.labelZh : op.labelEn}
                  </option>
                ))}
              </select>
              <button
                type="button"
                disabled={!selectedOperator}
                onClick={() => selectedOperator && addOperator(selectedOperator)}
                className="h-9 px-3 rounded-md bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[12px] font-semibold flex items-center gap-1 shrink-0 disabled:opacity-45 cursor-pointer transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>{lang === 'zh' ? '加入' : 'Add'}</span>
              </button>
            </div>
          </label>

          {/* Selected Operator Specification Preview Card */}
          {selectedOperator && (
            <div className="p-3 rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-900/50 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[12.5px] font-semibold text-slate-900 dark:text-slate-100 truncate">
                  {lang === 'zh' ? selectedOperator.labelZh : selectedOperator.labelEn}
                </span>
                <span className="font-mono text-[10px] uppercase text-blue-600 dark:text-cyan-400 shrink-0">
                  {selectedOperator.modality}
                </span>
              </div>
              <div className="font-mono text-[10.5px] text-slate-500">
                {selectedOperator.op} · {selectedOperator.inputs.join(', ')} →{' '}
                {selectedOperator.outputs.join(', ')}
              </div>
              <p className="text-[11.5px] leading-relaxed text-slate-600 dark:text-slate-400">
                {lang === 'zh' ? selectedOperator.descZh : selectedOperator.descEn}
              </p>
              {selectedOperator.params.length > 0 && (
                <div className="pt-1.5 border-t border-slate-200/80 dark:border-slate-800 text-[10.5px] font-mono text-slate-500">
                  {lang === 'zh' ? '可调参数: ' : 'Params: '}
                  {selectedOperator.params
                    .map((p) => (lang === 'zh' ? p.labelZh : p.labelEn))
                    .join(' · ')}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Right Column: Active Pipeline DAG / Step Composer */}
      <div className="lg:col-span-8 overflow-y-auto p-6 space-y-5">
        <div className="flex flex-wrap items-start justify-between gap-4 pb-4 border-b border-slate-200 dark:border-slate-800">
          <div>
            <h1 className="text-[18px] font-display font-semibold text-slate-900 dark:text-slate-100">
              {t('wf_title', lang)}
            </h1>
            <p className="mt-1 text-[13px] text-slate-500 dark:text-slate-400">
              {t('wf_subtitle', lang)}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.vexflow"
              onChange={handleImportFile}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="h-8 px-3 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12px] text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 flex items-center gap-1.5 cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>{t('wf_import', lang)}</span>
            </button>
            {!saveOpen ? (
              <button
                type="button"
                onClick={() => {
                  setSaveName(lang === 'zh' ? `${recipe.nameZh} (自定义)` : `${recipe.nameEn} (Custom)`);
                  setSaveOpen(true);
                }}
                className="h-8 px-3 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12px] text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 flex items-center gap-1.5 cursor-pointer"
              >
                <BookmarkPlus className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
                <span>{lang === 'zh' ? '另存工作流' : 'Save As…'}</span>
              </button>
            ) : (
              <form
                ref={saveFormRef}
                onSubmit={(e) => {
                  e.preventDefault();
                  if (!saveName.trim()) return;
                  onSaveCustomWorkflow(saveName.trim(), recipe);
                  setSaveOpen(false);
                }}
                className="flex items-center gap-1.5"
              >
                <input
                  type="text"
                  autoFocus
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  placeholder={lang === 'zh' ? '输入工作流名称…' : 'Workflow name…'}
                  className="h-8 w-44 px-2 rounded-md border border-blue-500 dark:border-cyan-500 bg-white dark:bg-slate-900 text-[12px] text-slate-900 dark:text-slate-100 focus:outline-none"
                />
                <button
                  type="submit"
                  className="h-8 px-2.5 rounded-md bg-blue-600 dark:bg-cyan-500 text-white dark:text-slate-950 text-[12px] font-semibold flex items-center gap-1 cursor-pointer"
                >
                  <Check className="w-3.5 h-3.5" />
                  <span>{lang === 'zh' ? '保存' : 'Save'}</span>
                </button>
                <button
                  type="button"
                  onClick={() => setSaveOpen(false)}
                  className="h-8 w-8 rounded-md border border-slate-300 dark:border-slate-700 text-slate-500 flex items-center justify-center cursor-pointer"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </form>
            )}
            <button
              type="button"
              onClick={onExportVexflow}
              className="h-8 px-3 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12px] text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 flex items-center gap-1.5 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>{t('wf_export', lang)}</span>
            </button>
            <button
              type="button"
              onClick={() => onApplyAndRun(recipe)}
              className="h-8 px-3.5 rounded-md bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 text-white dark:text-slate-950 text-[12px] font-semibold flex items-center gap-1.5 cursor-pointer"
            >
              <Play className="w-3.5 h-3.5" />
              <span>{t('wf_apply_run', lang)}</span>
            </button>
          </div>
        </div>

        {importError && (
          <div className="p-3 rounded-md border border-rose-500/40 bg-rose-500/10 text-[12px] text-rose-600 dark:text-rose-400">
            {importError}
          </div>
        )}

        {/* AI Natural-Language Workflow Edit Bar */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!aiPrompt.trim() || aiLoading) return;
            onAiEditWorkflow(aiPrompt.trim());
            setAiPrompt('');
          }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            value={aiPrompt}
            onChange={(e) => setAiPrompt(e.target.value)}
            placeholder={
              lang === 'zh'
                ? '自然语言改写工作流，例如：在颗粒分析前加入去条纹并启用 UNet 模型分割…'
                : 'Modify workflow in natural language, e.g. Add destriping and enable UNet segmentation…'
            }
            className="flex-1 h-9 px-3 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-[#0E1320] text-[13px] text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:border-blue-500 dark:focus:border-cyan-500"
          />
          <button
            type="submit"
            disabled={aiLoading || !aiPrompt.trim()}
            className="h-9 px-3.5 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 text-[12.5px] font-medium text-slate-800 dark:text-slate-100 flex items-center gap-1.5 disabled:opacity-45 cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
            <span>{aiLoading ? '...' : lang === 'zh' ? 'AI 编排调整' : 'AI Compose'}</span>
          </button>
        </form>

        {/* View Mode Switcher (DAG Visual Graph vs Full Parameter List) */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="inline-flex items-center gap-1 p-1 rounded-md bg-slate-200/70 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
            <button
              type="button"
              onClick={() => setViewMode('dag')}
              className={`h-7 px-3 rounded text-[12px] font-medium flex items-center gap-1.5 whitespace-nowrap cursor-pointer transition-colors ${
                viewMode === 'dag'
                  ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
            >
              <GitBranch className="w-3.5 h-3.5 text-blue-600 dark:text-cyan-400" />
              <span>{lang === 'zh' ? 'DAG 流程图谱' : 'DAG Graph'}</span>
            </button>
            <button
              type="button"
              onClick={() => setViewMode('list')}
              className={`h-7 px-3 rounded text-[12px] font-medium flex items-center gap-1.5 whitespace-nowrap cursor-pointer transition-colors ${
                viewMode === 'list'
                  ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
            >
              <List className="w-3.5 h-3.5" />
              <span>{lang === 'zh' ? '参数列表' : 'Step List'}</span>
            </button>
          </div>
          <div className="text-[11.5px] font-mono text-slate-500">
            {recipe.nameZh && lang === 'zh' ? recipe.nameZh : recipe.nameEn} · {recipe.steps.length}{' '}
            {lang === 'zh' ? '个算子节点' : 'nodes'}
          </div>
        </div>

        {/* Visual DAG Graph View or Ordered Parameter List */}
        {viewMode === 'dag' ? (
          <WorkflowDagGraph
            lang={lang}
            recipe={recipe}
            models={models}
            selectedStepId={selectedNodeId}
            onSelectStep={setSelectedNodeId}
            onReorderSteps={reorderSteps}
            onToggleStep={toggleStep}
            onRemoveStep={removeStep}
            onUpdateParam={updateParam}
            onUpdateModelRef={updateModelRef}
          />
        ) : (
        <div className="space-y-3">
          {recipe.steps.map((step, idx) => {
            const spec = OPERATOR_CATALOG.find((o) => o.op === step.op);
            return (
              <div
                key={step.id}
                className={`rounded-lg border transition-colors ${
                  step.enabled
                    ? 'border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E1320]'
                    : 'border-slate-200/70 dark:border-slate-800/60 bg-slate-100/60 dark:bg-slate-900/30 opacity-65'
                }`}
              >
                <div className="px-4 py-3 flex items-center justify-between gap-3 border-b border-slate-100 dark:border-slate-800/80">
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="font-mono text-[12px] font-semibold text-blue-600 dark:text-cyan-400 w-5">
                      {String(idx + 1).padStart(2, '0')}
                    </span>
                    <div className="min-w-0">
                      <div className="text-[13.5px] font-semibold text-slate-900 dark:text-slate-100 truncate">
                        {spec ? (lang === 'zh' ? spec.labelZh : spec.labelEn) : step.op}
                      </div>
                      <div className="font-mono text-[11px] text-slate-500">
                        {step.op} · {spec ? `${spec.inputs.join(',')} → ${spec.outputs.join(',')}` : ''}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-1.5 shrink-0">
                    <label className="flex items-center gap-1.5 text-[11.5px] text-slate-600 dark:text-slate-400 mr-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={step.enabled}
                        onChange={() => toggleStep(step.id)}
                        className="accent-cyan-500"
                      />
                      <span>{step.enabled ? (lang === 'zh' ? '启用' : 'On') : (lang === 'zh' ? '跳过' : 'Skip')}</span>
                    </label>
                    <button
                      type="button"
                      disabled={idx === 0}
                      onClick={() => moveStep(idx, -1)}
                      className="w-7 h-7 rounded border border-slate-200 dark:border-slate-700 flex items-center justify-center text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 disabled:opacity-30 cursor-pointer"
                    >
                      <ArrowUp className="w-3.5 h-3.5" />
                    </button>
                    <button
                      type="button"
                      disabled={idx === recipe.steps.length - 1}
                      onClick={() => moveStep(idx, 1)}
                      className="w-7 h-7 rounded border border-slate-200 dark:border-slate-700 flex items-center justify-center text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 disabled:opacity-30 cursor-pointer"
                    >
                      <ArrowDown className="w-3.5 h-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => removeStep(step.id)}
                      className="w-7 h-7 rounded border border-slate-200 dark:border-slate-700 flex items-center justify-center text-slate-500 hover:text-rose-500 cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {/* Inline step configuration */}
                <div className="px-4 py-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-[12px]">
                  {step.op === 'model.segment' && (
                    <label className="md:col-span-2 space-y-1">
                      <span className="text-[11px] text-slate-500 dark:text-slate-400 block">
                        {lang === 'zh' ? 'Trained Model 权重绑定' : 'Trained Model Binding'}
                      </span>
                      <select
                        value={step.modelRef ?? models[0]?.ref ?? ''}
                        onChange={(e) => updateModelRef(step.id, e.target.value)}
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

                  {spec?.params.map((p) => {
                    const val = step.params[p.key] ?? p.defaultValue;
                    return (
                      <div key={p.key} className="space-y-1">
                        <div className="flex justify-between text-[11.5px]">
                          <span className="text-slate-600 dark:text-slate-300">
                            {lang === 'zh' ? p.labelZh : p.labelEn}
                          </span>
                          {p.type === 'number' && (
                            <span className="font-mono text-slate-500">
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
                            onChange={(e) => updateParam(step.id, p.key, Number(e.target.value))}
                            className="w-full accent-blue-600 dark:accent-cyan-400"
                          />
                        )}
                        {p.type === 'boolean' && (
                          <label className="inline-flex items-center gap-2 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={Boolean(val)}
                              onChange={(e) => updateParam(step.id, p.key, e.target.checked)}
                              className="accent-cyan-500"
                            />
                            <span className="text-slate-700 dark:text-slate-300">
                              {Boolean(val) ? 'true' : 'false'}
                            </span>
                          </label>
                        )}
                        {p.type === 'choice' && p.choices && (
                          <select
                            value={String(val)}
                            onChange={(e) => updateParam(step.id, p.key, e.target.value)}
                            className="w-full h-7 px-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12px] text-slate-900 dark:text-slate-100"
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
                </div>
              </div>
            );
          })}
        </div>
        )}
      </div>
    </div>
  );
}
