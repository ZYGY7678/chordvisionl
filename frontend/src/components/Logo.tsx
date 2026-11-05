/**
 * The mark is four bars at four heights: a chromagram frame, which is
 * exactly what the engine reads a chord out of. It is the product's own
 * internal picture, not a borrowed music cliche.
 */
export function Logo({ size = 28 }: { size?: number }) {
  const bars = [
    { x: 7, y: 13, h: 6 },
    { x: 12.5, y: 8, h: 16 },
    { x: 18, y: 11, h: 10 },
    { x: 23.5, y: 15, h: 2 },
  ];
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" className="fill-raised" />
      <g className="fill-glow">
        {bars.map((b, i) => (
          <rect key={i} x={b.x} y={b.y} width="3" height={b.h} rx="1.5" />
        ))}
      </g>
    </svg>
  );
}
