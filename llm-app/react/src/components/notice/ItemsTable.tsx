import type { NoticeItem } from "../../types"

export function ItemsTable({ items }: { items: NoticeItem[] }) {
  if (items.length === 0) {
    return <p className="text-sm text-zinc-400">등록된 품목 정보가 없습니다.</p>
  }

  const columns = Array.from(new Set(items.flatMap((item) => Object.keys(item))))

  return (
    <div className="overflow-x-auto rounded-xl border border-zinc-200">
      <table className="min-w-full divide-y divide-zinc-200 text-sm">
        <thead className="bg-zinc-50">
          <tr>
            {columns.map((col) => (
              <th key={col} className="whitespace-nowrap px-3 py-2 text-left text-xs font-medium text-zinc-500">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-100 bg-white">
          {items.map((item, i) => (
            <tr key={i}>
              {columns.map((col) => (
                <td key={col} className="whitespace-nowrap px-3 py-2 text-zinc-700">
                  {item[col] != null ? String(item[col]) : "-"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
