import { useNavigate } from "react-router-dom"
import { useAppState } from "../../context/appStateStore"
import { MenuIcon } from "../common/Icons"

export function Topbar() {
  const navigate = useNavigate()
  const { selectedAgent, sidebarOpen, setSidebarOpen } = useAppState()

  return (
    <header className="flex items-center gap-3 border-b border-zinc-200 bg-white/80 px-4 py-3 backdrop-blur lg:px-6">
      <button
        type="button"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-100"
        aria-label={sidebarOpen ? "사이드바 닫기" : "사이드바 열기"}
      >
        <MenuIcon className="size-5" />
      </button>
      <button
        type="button"
        onClick={() => navigate("/")}
        className="truncate text-sm font-semibold text-zinc-800 hover:text-brand-600"
      >
        {selectedAgent ? selectedAgent.name : "공공조달 어시스턴트"}
      </button>
    </header>
  )
}
