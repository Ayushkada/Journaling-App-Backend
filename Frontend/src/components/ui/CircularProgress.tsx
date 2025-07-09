interface CircularProgressProps {
  value: number; // 0–100
  size?: number; // diameter in pixels (defaults to 90)
  strokeWidth?: number; // thickness of the ring (defaults to 6)
}

const CircularProgress: React.FC<CircularProgressProps> = ({
  value,
  size = 90,
  strokeWidth = 6,
}) => {
  // Keep the SVG viewBox fixed at 72×72 so it scales nicely
  const radius = 28;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = 2 * Math.PI * normalizedRadius;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="w-full h-full" viewBox="0 0 72 72">
        {/* background ring */}
        <circle
          cx="36"
          cy="36"
          r={normalizedRadius}
          stroke="#E5E7EB"
          strokeWidth={strokeWidth}
          fill="none"
        />
        {/* progress ring */}
        <circle
          cx="36"
          cy="36"
          r={normalizedRadius}
          stroke="#34D1BF"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="text-sm font-bold text-foreground">{value}%</span>
      </div>
    </div>
  );
};

export default CircularProgress;
