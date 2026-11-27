import { useEffect, useRef, type RefObject } from 'react';

/**
 * Invokes `onClose` when the user clicks/taps outside all provided element refs
 * or presses the Escape key while `active` is true.
 */
export function useClickOutside(
  active: boolean,
  refs: Array<RefObject<HTMLElement | null>>,
  onClose: () => void
) {
  const refsLatest = useRef(refs);
  const onCloseLatest = useRef(onClose);
  refsLatest.current = refs;
  onCloseLatest.current = onClose;

  useEffect(() => {
    if (!active) return;

    const handlePointerDown = (event: MouseEvent | PointerEvent) => {
      const target = event.target as Node | null;
      if (!target) return;
      for (const ref of refsLatest.current) {
        if (ref.current && ref.current.contains(target)) {
          return;
        }
      }
      onCloseLatest.current();
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onCloseLatest.current();
      }
    };

    document.addEventListener('pointerdown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [active]);
}
