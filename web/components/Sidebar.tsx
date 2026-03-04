"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Dashboard", icon: "⊞" },
  { href: "/model-performance", label: "Model Performance", icon: "◈" },
  { href: "/confusion-matrices", label: "Confusion Matrices", icon: "⊟" },
  { href: "/eda", label: "EDA Explorer", icon: "◉" },
  { href: "/dataset", label: "Dataset Overview", icon: "⊡" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-[220px] bg-card border-r border-border flex flex-col z-50">
      <div className="p-4 border-b border-border">
        <h1 className="text-sm font-bold text-foreground tracking-tight">Medical Insurance</h1>
        <p className="text-xs text-muted mt-0.5">ML Dashboard</p>
      </div>
      <nav className="flex-1 p-2 space-y-0.5">
        {links.map(({ href, label, icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-md text-sm transition-colors ${
                active
                  ? "bg-accent/15 text-accent font-medium"
                  : "text-muted hover:text-foreground hover:bg-card-hover"
              }`}
            >
              <span className="text-base">{icon}</span>
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t border-border">
        <p className="text-xs text-muted">ECS 171 Project</p>
      </div>
    </aside>
  );
}
