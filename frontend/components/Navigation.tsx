"use client";

import { useState } from "react";
import { Brain, Menu, Satellite, Settings, UserCircle, X } from "lucide-react";

const navItems = [
  { label: "MISSION", href: "#mission" },
  { label: "SUPER RESOLUTION", href: "#processing" },
  { label: "SATELLITE DATA", href: "#satellite" },
  { label: "MAP", href: "#map" },
  { label: "ANALYTICS", href: "#analytics" },
  { label: "TECHNOLOGY", href: "#technology" },
];

export default function Navigation({ online }: { online: boolean }) {
  const [open, setOpen] = useState(false);

  return (
    <header className="glass-nav fixed inset-x-0 top-0 z-50">
      <nav className="mx-auto flex h-16 max-w-[1480px] items-center justify-between px-4 sm:px-6 lg:px-8">
        <a href="#hero" className="flex items-center gap-3" aria-label="GeoSR AI home">
          <span className="relative flex h-10 w-10 items-center justify-center border border-cyan-signal/40 bg-cyan-signal/10 shadow-glow">
            <Satellite className="h-5 w-5 text-cyan-signal" aria-hidden="true" />
            <Brain className="absolute -right-1 -top-1 h-3.5 w-3.5 text-orbit-green" aria-hidden="true" />
            <span className="absolute inset-1 rounded-full border border-white/10" />
          </span>
          <span>
            <span className="font-display block text-sm font-semibold uppercase tracking-[0.2em] text-white">GeoSR AI</span>
            <span className="font-telemetry block text-[10px] uppercase tracking-[0.24em] text-muted">Mission Control</span>
          </span>
        </a>

        <div className="hidden items-center gap-1 lg:flex">
          {navItems.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="font-telemetry px-3 py-2 text-[11px] uppercase tracking-[0.18em] text-slate-300 transition hover:text-cyan-signal"
            >
              {item.label}
            </a>
          ))}
        </div>

        <div className="hidden items-center gap-4 lg:flex">
          <div className="font-telemetry flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-300">
            <span className={online ? "status-dot" : "warning-dot"} />
            {online ? "SYSTEM ONLINE" : "API STANDBY"}
          </div>
          <button
            className="flex h-9 w-9 items-center justify-center border border-white/10 bg-white/5 text-slate-200 transition hover:border-cyan-signal/40 hover:text-cyan-signal"
            aria-label="Open settings"
            title="Settings"
          >
            <Settings className="h-4 w-4" aria-hidden="true" />
          </button>
          <button
            className="flex h-9 w-9 items-center justify-center border border-cyan-signal/20 bg-cyan-signal/10 text-cyan-signal transition hover:bg-cyan-signal/15"
            aria-label="Open user profile"
            title="User profile"
          >
            <UserCircle className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        <button
          className="flex h-10 w-10 items-center justify-center border border-white/10 bg-white/5 text-slate-100 lg:hidden"
          aria-label="Open navigation"
          onClick={() => setOpen(true)}
        >
          <Menu className="h-5 w-5" aria-hidden="true" />
        </button>
      </nav>

      {open ? (
        <div className="fixed inset-0 z-50 bg-space-950/95 px-4 py-5 backdrop-blur-xl lg:hidden">
          <div className="flex items-center justify-between">
            <span className="font-display text-sm font-semibold uppercase tracking-[0.24em] text-white">GeoSR AI</span>
            <button
              className="flex h-10 w-10 items-center justify-center border border-white/10 bg-white/5"
              aria-label="Close navigation"
              onClick={() => setOpen(false)}
            >
              <X className="h-5 w-5" aria-hidden="true" />
            </button>
          </div>
          <div className="mt-10 grid gap-2">
            {navItems.map((item) => (
              <a
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                className="font-telemetry border border-white/10 bg-white/[0.03] px-4 py-4 text-xs uppercase tracking-[0.2em] text-slate-200"
              >
                {item.label}
              </a>
            ))}
          </div>
        </div>
      ) : null}
    </header>
  );
}
