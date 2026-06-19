import { BrowserRouter, Route, Routes } from "react-router-dom"
import { AppStateProvider } from "./context/AppStateContext"
import { Shell } from "./components/layout/Shell"
import { AgentPickerPage } from "./pages/AgentPickerPage"
import { AgentRouterPage } from "./pages/AgentRouterPage"

function App() {
  return (
    <AppStateProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Shell />}>
            <Route path="/" element={<AgentPickerPage />} />
            <Route path="/agent/:mode" element={<AgentRouterPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppStateProvider>
  )
}

export default App
