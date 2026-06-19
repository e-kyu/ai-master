export function EmptyState({ title, description }: { title: string; description?: string }) {
  return (
    <div className="rounded-2xl border border-dashed border-zinc-300 bg-zinc-50/60 px-6 py-12 text-center">
      <p className="text-base font-semibold text-zinc-600">{title}</p>
      {description && <p className="mt-1.5 text-sm text-zinc-400">{description}</p>}
    </div>
  )
}
