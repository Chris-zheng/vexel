/**
 * Does a workflow fit the image it runs on? Every workflow gives numbers on any image - a top-down
 * line pattern measured as a cross-section gives "75 layers" - so the results say so when the
 * workflow and what the image was recognised as disagree. Unknown images are not judged.
 */

import type { Specimen } from '../types/vexel';

type SubType = Specimen['subType'];

const FITS: Record<string, { subTypes: SubType[]; scenes: string[] }> = {
  droplet_encapsulation: { subTypes: ['droplet_encapsulation', 'droplet_fluorescence'], scenes: ['microfluidics_droplet'] },
  droplet_fluorescence: { subTypes: ['droplet_fluorescence', 'droplet_encapsulation'], scenes: ['microfluidics_droplet'] },
  sem_lines: { subTypes: ['sem_lines'], scenes: ['sem_metrology'] },
  sem_holes: { subTypes: ['sem_holes', 'sem_pores'], scenes: ['sem_metrology'] },
  sem_particles: { subTypes: ['sem_particles', 'sem_pores'], scenes: ['sem_metrology'] },
  sem_stacked_particles: { subTypes: ['sem_particles'], scenes: ['sem_metrology'] },
  sem_fibres: { subTypes: ['sem_fibres'], scenes: ['sem_metrology'] },
  sem_multilayer: { subTypes: ['sem_multilayer'], scenes: ['sem_metrology'] },
  sem_curved_multilayer: { subTypes: ['sem_multilayer'], scenes: ['sem_metrology'] },
  nested_cell_nucleus: { subTypes: ['kidney_nested_cells', 'cells_count'], scenes: ['cells'] },
  cell_segmentation: { subTypes: ['cells_count', 'kidney_nested_cells'], scenes: ['cells'] },
  pcb_inspection: { subTypes: ['pcb_traces'], scenes: ['pcb_board'] },
};

const SUBTYPE_NAMES: Partial<Record<SubType, [string, string]>> = {
  droplet_encapsulation: ['微流控液滴', 'microfluidic droplets'],
  droplet_fluorescence: ['荧光液滴', 'fluorescent droplets'],
  cells_count: ['细胞', 'cells'],
  pcb_traces: ['PCB 走线', 'PCB traces'],
  sem_lines: ['SEM 俯视线条', 'SEM top-down lines'],
  sem_holes: ['SEM 孔阵列', 'SEM hole array'],
  sem_particles: ['SEM 颗粒', 'SEM particles'],
  sem_pores: ['SEM 多孔', 'SEM pores'],
  sem_fibres: ['SEM 纤维', 'SEM fibres'],
  sem_multilayer: ['SEM 多层截面', 'SEM layered cross-section'],
  kidney_nested_cells: ['组织切片（细胞与细胞核）', 'tissue section (cells and nuclei)'],
};

/** null when the workflow fits or the image is not recognised; else what the image looks like.
 * Built-in samples are judged by their exact kind. Uploaded images only by their scene (SEM,
 * cells, droplets, PCB): within SEM the kind of an upload (lines, holes, particles, layers) is a
 * guess from its file name and simple image statistics, with "particles" as the fallback, and a
 * wrong warning on a right choice would teach people to ignore the warning. */
export function workflowMismatch(specimen: Specimen, recipeId: string, lang: 'zh' | 'en'): string | null {
  const fit = FITS[recipeId];
  if (!fit) return null; // generic workflows fit anything
  const zh = lang === 'zh';
  if (!specimen.importReport && specimen.subType !== 'uploaded' && specimen.subType !== 'unknown') {
    if (fit.subTypes.includes(specimen.subType)) return null;
    const name = SUBTYPE_NAMES[specimen.subType];
    return name ? name[zh ? 0 : 1] : specimen.subType;
  }
  const rep = specimen.importReport;
  if (!rep || !rep.recognized || rep.sceneType === 'unknown') return null;
  if (fit.scenes.includes(rep.sceneType)) return null;
  return zh ? rep.sceneLabelZh : rep.sceneLabelEn;
}
