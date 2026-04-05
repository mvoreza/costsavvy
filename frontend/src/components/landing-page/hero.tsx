"use client";
import React, { useEffect, useState } from "react";
import PriceSearch from "./PriceSearch";

interface HeroProps {
  tagline: string;
  rotatingWords: string[];
  commonProcedures: string[];
}

export default function Hero({ tagline, rotatingWords, commonProcedures }: HeroProps) {
  const [index, setIndex] = useState(0);
  const [isFading, setIsFading] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setIsFading(true);
      setTimeout(() => {
        setIndex((prev) => (prev === rotatingWords.length - 1 ? 0 : prev + 1));
        setIsFading(false);
      }, 400);
    }, 2000);
    return () => clearInterval(interval);
  }, [rotatingWords]);

  return (
    <main className="px-6 sm:px-12 py-10 sm:py-16">
      {/* Headline */}
      <h1 className="text-[#403B3D] text-3xl md:text-5xl lg:text-6xl font-bold font-serif leading-tight mb-4">
        {tagline} <span className="md:hidden">for</span>
        <br />
        <span className="flex items-center space-x-4">
          <span className="hidden sm:inline-block">for</span>
          <span className={`text-[#8C2F5D] transition-opacity duration-400 ${isFading ? "opacity-0" : "opacity-100"}`}>
            {rotatingWords[index]}
          </span>
        </span>
      </h1>

      {/* Quick-select chips from Sanity commonProcedures */}
      {commonProcedures.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-6">
          {commonProcedures.map((p) => (
            <span key={p} className="px-3 py-1 rounded-full bg-[#f3e8ef] text-[#8C2F5D] text-sm font-medium">
              {p}
            </span>
          ))}
        </div>
      )}

      {/* AI Price Search — replaces old SearchBar */}
      <PriceSearch />
    </main>
  );
}
