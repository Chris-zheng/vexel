import type { Lang } from '../types/vexel';

const DICT: Record<string, [string, string]> = {
  // Navigation
  nav_workbench: ['工作台', 'Workbench'],
  nav_workflow: ['工作流编排', 'Workflow'],
  nav_models: ['模型中心', 'Models'],
  nav_studio3d: ['3D 实验室', '3D Lab'],
  nav_batch: ['批量分析', 'Batch'],
  btn_export: ['导出报告', 'Export'],
  btn_run: ['运行分析', 'Run'],
  btn_recompute: ['重算', 'Recompute'],

  // Samples & Upload
  sample_select: ['标准样本', 'Specimens'],
  upload_image: ['导入图像', 'Import Image'],
  modality_droplet: ['微流控液滴', 'Microfluidics'],
  modality_sem: ['SEM 扫描电镜', 'SEM Metrology'],

  // Calibration Banner
  cal_confirmed: ['标定已确认', 'Calibrated'],
  cal_pending: ['待确认标定', 'Pending confirmation'],
  cal_none: ['未标定（像素单位）', 'Uncalibrated (px)'],
  cal_confirm_btn: ['确认标定', 'Confirm'],
  cal_edit_btn: ['编辑标定', 'Calibrate'],
  cal_source_metadata: ['仪器元数据', 'Instrument metadata'],
  cal_source_databar: ['信息栏比例尺识别', 'Data bar OCR'],
  cal_source_manual: ['手动标定', 'Manual'],
  cal_source_none: ['无标定', 'None'],

  // Result-First Plan Card
  plan_title: ['自动识别与规划', 'Auto Recognition & Plan'],
  plan_ready: ['试运行通过', 'Trial Passed'],
  plan_check: ['通过 · 有提示', 'Passed with Checks'],
  plan_needs_user: ['需要人工确认', 'Needs Review'],
  plan_accept: ['采用方案', 'Accept'],
  plan_adjust: ['调整工作流', 'Customize'],
  plan_manual: ['手动设置', 'Manual'],
  plan_goal_placeholder: ['输入目标或指令，如：测线宽 LER 并去条纹 / 统计细胞包裹率…', 'Describe goal, e.g. Measure CD & LER with destriping / Count encapsulated cells…'],
  plan_ai_btn: ['智能规划', 'Auto Plan'],
  plan_send_image: ['允许发送降采样诊断图至 AI', 'Allow sending downsampled diagnostic image to AI'],

  // Canvas HUD & Tools
  tool_pan: ['选择 / 平移', 'Select / Pan'],
  tool_roi: ['手动框选 ROI', 'Draw ROI'],
  tool_calibrate_line: ['手动描线标定比例尺', 'Calibrate Scale Bar'],
  tool_line: ['标尺测距', 'Ruler'],
  tool_profile: ['线剖面', 'Line Profile'],
  tool_measure_object: ['智能点选测物体 (长宽/面积)', 'Smart Click Measure (L/W/Area)'],
  tool_smart_gt: ['SAM 智能点选标注 GT', 'Smart Click GT (SAM)'],
  tool_boundary_refine: ['SAM 边界精修 (悬停预览/确认应用)', 'Boundary Refine (SAM Hover & Confirm)'],
  tool_prompt_pos: ['点选分割 (+)', 'Prompt (+)'],
  tool_prompt_neg: ['排除区域 (-)', 'Prompt (-)'],
  tool_brush: ['多类画笔标注', 'Multi-Class Brush'],
  canvas_fit: ['适应窗口', 'Fit'],
  canvas_reset_lut: ['重置对比度', 'Reset LUT'],
  canvas_layers: ['叠加图层', 'Overlays'],
  canvas_opacity: ['不透明度', 'Opacity'],
  canvas_step_preview: ['当前预览步骤', 'Previewing Step'],
  canvas_source: ['原始图像', 'Source Image'],
  canvas_fluo_channel: ['荧光通道', 'Fluorescence'],

  // Inspector Tabs
  tab_results: ['结果与图表', 'Results'],
  tab_params: ['高级参数', 'Parameters'],
  tab_review: ['复核与纠错', 'Review & Correct'],
  tab_provenance: ['溯源与方法', 'Provenance'],
  params_reset: ['恢复默认', 'Reset Defaults'],
  params_modified: ['已修改', 'Modified'],

  // Review & Active Learning
  review_all: ['全部目标', 'All Objects'],
  review_flagged: ['可疑目标', 'Flagged Only'],
  review_exclude: ['排除目标', 'Exclude'],
  review_restore: ['恢复目标', 'Restore'],
  review_send_train: ['将纠错样本送入模型微调', 'Send Corrections to Model Fine-Tuning'],
  review_no_corrections: ['尚无人工修正记录', 'No human corrections yet'],

  // Workflow Studio
  wf_title: ['可视化自定义工作流', 'Custom Workflow Studio'],
  wf_subtitle: ['拖拽/调整步骤顺序，替换 Trained Model 节点，冻结为可复现的 .vexflow 模板', 'Compose deterministic operators and trained models into frozen, reproducible .vexflow pipelines'],
  wf_add_step: ['添加算子 / 模型节点', 'Add Operator / Model'],
  wf_import: ['导入 .vexflow', 'Import .vexflow'],
  wf_export: ['保存 .vexflow', 'Export .vexflow'],
  wf_apply_run: ['应用并运行工作流', 'Apply & Execute'],
  wf_preset_recipes: ['内置工作流模板', 'Built-in Presets'],

  // Model Hub
  models_title: ['Trained Model 图形检测与主动学习中心', 'Trained Model Hub & Active Learning'],
  models_subtitle: ['所有模型强制校验权重 SHA-256 与训练集许可证，支持大图分块推理与画笔/纠错本地微调', 'SHA-256 verified weights, dataset license audit, tiled inference, and human-in-the-loop local fine-tuning'],
  models_bind: ['绑定至当前工作流', 'Bind to Workflow'],
  models_bound: ['已绑定', 'Active in Workflow'],
  models_train_title: ['本地多类模型微调（Active Learning）', 'Local Multi-Class Model Training'],
  models_train_btn: ['训练并留出集评估', 'Train & Evaluate Hold-Out'],
  models_publish_btn: ['发布为可溯源模型版本', 'Publish Versioned Model'],
  models_commercial_ok: ['可商用', 'Commercial OK'],
  models_commercial_no: ['仅科研非商用', 'Research Only'],

  // Batch Center
  batch_title: ['统一多格式批量分析与分组统计', 'Unified Batch Processing & Experiment Statistics'],
  batch_subtitle: ['微流控与 SEM 共用实验设计表，以重复（Replicate）为单位执行统计检验并聚合异常样本复核', 'Shared design table across Microfluidics and SEM with replicate-level statistics and QC review queue'],
  batch_add_samples: ['载入演示组', 'Load Demo Cohort'],
  batch_upload_files: ['批量添加图像', 'Add Image Files'],
  batch_run_all: ['运行全部批处理', 'Run Batch Queue'],
  batch_export_xlsx: ['导出多 Sheet Excel', 'Export Multi-Sheet .xlsx'],
  batch_export_reproduce: ['导出 reproduce.py', 'Export reproduce.py'],
  batch_abnormal_queue: ['异常 / 低置信复核队列', 'Flagged & QC Review Queue'],
  batch_jump_review: ['打开画布复核', 'Inspect on Canvas'],

  // Backend / Settings
  engine_web: ['内置实时内核', 'Built-in Web Kernel'],
  engine_python: ['本地 Python 引擎', 'Local Python Engine'],
  engine_offline: ['离线合规模式', 'Offline Policy'],
};

export function t(key: string, lang: Lang): string {
  const pair = DICT[key];
  if (!pair) return key;
  return lang === 'en' ? pair[1] : pair[0];
}
