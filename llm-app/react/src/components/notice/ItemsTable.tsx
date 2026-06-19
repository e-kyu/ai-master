import type { NoticeItem } from "../../types"

const COLUMN_LABELS: Record<string, string> = {
  itemNo: "번호",
  itemName: "품목명",
  standard: "규격",
  unit: "단위",
  quantity: "수량",
  unitPrice: "단가",
  amount: "금액",
}

const NUMERIC_COLUMNS = new Set(["quantity", "unitPrice", "amount"])

function formatValue(col: string, value: unknown) {
  if (value == null || value === "") return "-"
  if (NUMERIC_COLUMNS.has(col) && typeof value === "number") return value.toLocaleString()
  return String(value)
}

export function ItemsTable({ items }: { items: NoticeItem[] }) {
  if (items.length === 0) {
    return <p className="text-sm text-zinc-400">등록된 품목 정보가 없습니다.</p>
  }

  const columns = Array.from(new Set(items.flatMap((item) => Object.keys(item))))
  const totalQuantity = items.reduce((sum, item) => (typeof item.quantity === "number" ? sum + item.quantity : sum), 0)
  const totalAmount = items.reduce((sum, item) => (typeof item.amount === "number" ? sum + item.amount : sum), 0)

  return (
    <div className="overflow-x-auto rounded-xl border border-zinc-200">
      <table className="min-w-full divide-y divide-zinc-200 text-sm">
        <thead className="bg-zinc-50">
          <tr>
            {columns.map((col) => (
              <th
                key={col}
                className={`whitespace-nowrap px-3 py-2 text-xs font-medium text-zinc-500 ${
                  NUMERIC_COLUMNS.has(col) ? "text-right" : "text-left"
                }`}
              >
                {COLUMN_LABELS[col] ?? col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-100 bg-white">
          {items.map((item, i) => (
            <tr key={i}>
              {columns.map((col) => (
                <td
                  key={col}
                  className={`whitespace-nowrap px-3 py-2 text-zinc-700 ${NUMERIC_COLUMNS.has(col) ? "text-right" : ""}`}
                >
                  {formatValue(col, item[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
        {(totalQuantity > 0 || totalAmount > 0) && (
          <tfoot className="border-t border-zinc-200 bg-zinc-50">
            <tr>
              {columns.map((col) => (
                <td key={col} className={`whitespace-nowrap px-3 py-2 text-xs font-semibold text-zinc-600 ${NUMERIC_COLUMNS.has(col) ? "text-right" : ""}`}>
                  {col === "itemName" ? "합계" : col === "quantity" && totalQuantity > 0 ? totalQuantity.toLocaleString() : col === "amount" && totalAmount > 0 ? totalAmount.toLocaleString() : ""}
                </td>
              ))}
            </tr>
          </tfoot>
        )}
      </table>
    </div>
  )
}
