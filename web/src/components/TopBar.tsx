import { useRef, useState } from 'react';
import {
  Download,
  FileCode2,
  FileJson,
  FileSpreadsheet,
  FileText,
  Image as ImageIcon,
  Table,
  ChevronDown,
} from 'lucide-react';
import type { AppView, Lang } from '../types/vexel';
import { t } from '../i18n/strings';
import { useClickOutside } from '../hooks/useClickOutside';

interface TopBarProps {
  lang: Lang;
  view: AppView;
  onChangeView: (view: AppView) => void;
  hasResult: boolean;
  onExportExcel: () => void;
  onExportCsv: () => void;
  onExportSvg: () => void;
  onExportHtml: () => void;
  onExportVexflow: () => void;
  onExportReproducePy: () => void;
}

const NAV_ITEMS: Array<{ id: AppView; key: string }> = [
  { id: 'workbench', key: 'nav_workbench' },
  { id: 'workflow', key: 'nav_workflow' },
  { id: 'models', key: 'nav_models' },
  { id: 'studio3d', key: 'nav_studio3d' },
  { id: 'batch', key: 'nav_batch' },
];

export function TopBar({
  lang,
  view,
  onChangeView,
  hasResult,
  onExportExcel,
  onExportCsv,
  onExportSvg,
  onExportHtml,
  onExportVexflow,
  onExportReproducePy,
}: TopBarProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useClickOutside(menuOpen, [menuRef], () => setMenuOpen(false));

  const runAction = (fn: () => void) => {
    setMenuOpen(false);
    fn();
  };

  return (
    <header className="h-12 shrink-0 border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0B0E17] px-4 flex items-center justify-between gap-4 select-none z-30">
      {/* Zone 1: Brand Wordmark */}
      <button
        type="button"
        onClick={() => onChangeView('workbench')}
        className="flex items-center gap-2.5 whitespace-nowrap shrink-0 group cursor-pointer focus:outline-none"
      >
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="none"
          aria-hidden="true"
          className="text-blue-600 dark:text-cyan-400 shrink-0"
        >
          <rect
            x="2.5"
            y="2.5"
            width="19"
            height="19"
            rx="3.5"
            stroke="currentColor"
            strokeWidth="1.6"
          />
          <path
            d="M7 7.5L12 16.5L17 7.5"
            stroke="currentColor"
            strokeWidth="1.9"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <circle cx="12" cy="8" r="1.4" fill="currentColor" />
        </svg>
        <span className="font-display font-semibold tracking-tight text-[15px] text-slate-900 dark:text-slate-100">
          Vexel
        </span>
      </button>

      {/* Zone 2: Primary Navigation Links */}
      <nav aria-label="Main navigation" className="flex items-center gap-1 whitespace-nowrap shrink-0">
        {NAV_ITEMS.map((item) => {
          const active = view === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onChangeView(item.id)}
              className={`relative px-3.5 h-9 rounded-md text-[13px] font-medium whitespace-nowrap shrink-0 transition-colors cursor-pointer ${
                active
                  ? 'text-blue-700 dark:text-cyan-300 bg-blue-50/90 dark:bg-cyan-500/12 font-semibold'
                  : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800/80'
              }`}
            >
              {t(item.key, lang)}
              {active && (
                <span className="absolute -bottom-1.5 left-2 right-2 h-[2px] bg-blue-600 dark:bg-cyan-400" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Zone 3: Single Primary Action (Export Menu) */}
      <div className="relative shrink-0" ref={menuRef}>
        <button
          type="button"
          disabled={!hasResult}
          onClick={() => setMenuOpen((o) => !o)}
          className="h-8 px-3.5 rounded-md bg-blue-600 hover:bg-blue-700 dark:bg-cyan-500 dark:hover:bg-cyan-400 dark:text-slate-950 text-white text-[13px] font-semibold shadow-xs flex items-center gap-1.5 whitespace-nowrap shrink-0 transition-colors disabled:opacity-40 cursor-pointer"
        >
          <Download className="w-3.5 h-3.5" />
          <span>{t('btn_export', lang)}</span>
          <ChevronDown className="w-3.5 h-3.5 opacity-80" />
        </button>

        {menuOpen && hasResult && (
          <div className="absolute right-0 mt-1.5 w-60 rounded-lg border border-slate-200 dark:border-slate-700/90 bg-white dark:bg-[#111827] shadow-lg py-1 z-50">
            <button
              type="button"
              onClick={() => runAction(onExportExcel)}
              className="w-full px-3 py-2 text-left text-[13px] text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 flex items-center gap-2.5 cursor-pointer"
            >
              <FileSpreadsheet className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
              <div className="min-w-0">
                <div className="font-medium truncate">
                  {lang === 'zh' ? '多 Sheet Excel (.xlsx)' : 'Multi-Sheet Excel (.xlsx)'}
                </div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400">
                  Summary · Objects · Provenance
                </div>
              </div>
            </button>
            <button
              type="button"
              onClick={() => runAction(onExportCsv)}
              className="w-full px-3 py-2 text-left text-[13px] text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 flex items-center gap-2.5 cursor-pointer"
            >
              <Table className="w-4 h-4 text-blue-600 dark:text-cyan-400 shrink-0" />
              <div className="min-w-0">
                <div className="font-medium truncate">
                  {lang === 'zh' ? '逐目标明细 CSV (UTF-8)' : 'Objects Table CSV (UTF-8)'}
                </div>
              </div>
            </button>
            <button
              type="button"
              onClick={() => runAction(onExportSvg)}
              className="w-full px-3 py-2 text-left text-[13px] text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 flex items-center gap-2.5 cursor-pointer"
            >
              <ImageIcon className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
              <div className="min-w-0">
                <div className="font-medium truncate">
                  {lang === 'zh' ? '矢量标注图 (.svg + 比例尺)' : 'Annotated Vector (.svg + Bar)'}
                </div>
              </div>
            </button>
            <button
              type="button"
              onClick={() => runAction(onExportHtml)}
              className="w-full px-3 py-2 text-left text-[13px] text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 flex items-center gap-2.5 cursor-pointer"
            >
              <FileText className="w-4 h-4 text-violet-600 dark:text-violet-400 shrink-0" />
              <div className="min-w-0">
                <div className="font-medium truncate">
                  {lang === 'zh' ? '可打印报告 (.html / PDF)' : 'Printable Report (.html / PDF)'}
                </div>
              </div>
            </button>

            <div className="my-1 border-t border-slate-200 dark:border-slate-800" />

            <button
              type="button"
              onClick={() => runAction(onExportVexflow)}
              className="w-full px-3 py-2 text-left text-[13px] text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 flex items-center gap-2.5 cursor-pointer"
            >
              <FileJson className="w-4 h-4 text-slate-500 dark:text-slate-400 shrink-0" />
              <div className="min-w-0">
                <div className="font-medium truncate">
                  {lang === 'zh' ? '冻结工作流模板 (.vexflow)' : 'Frozen Workflow (.vexflow)'}
                </div>
              </div>
            </button>
            <button
              type="button"
              onClick={() => runAction(onExportReproducePy)}
              className="w-full px-3 py-2 text-left text-[13px] text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 flex items-center gap-2.5 cursor-pointer"
            >
              <FileCode2 className="w-4 h-4 text-slate-500 dark:text-slate-400 shrink-0" />
              <div className="min-w-0">
                <div className="font-medium truncate">
                  {lang === 'zh' ? '复现脚本 (reproduce.py)' : 'Reproducibility Script (.py)'}
                </div>
              </div>
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
