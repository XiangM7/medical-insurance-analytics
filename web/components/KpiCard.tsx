"use client";

import AnimatedCounter from "./AnimatedCounter";

interface KpiCardProps {
  title: string;
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  subtitle?: string;
}

export default function KpiCard({
  title,
  value,
  decimals = 2,
  prefix = "",
  suffix = "",
  subtitle,
}: KpiCardProps) {
  return (
    <div className="bg-card border border-border rounded-lg p-4 hover:bg-card-hover transition-colors">
      <p className="text-xs text-muted uppercase tracking-wider mb-1">{title}</p>
      <p className="text-2xl font-bold text-foreground">
        <AnimatedCounter
          target={value}
          decimals={decimals}
          prefix={prefix}
          suffix={suffix}
        />
      </p>
      {subtitle && <p className="text-xs text-muted mt-1">{subtitle}</p>}
    </div>
  );
}
