import { API_BASE_URL, ApiError } from "./client"
import type { UploadResponse } from "../types"

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append("file", file)

  const res = await fetch(`${API_BASE_URL}/upload/`, {
    method: "POST",
    body: formData,
  })
  if (!res.ok) throw new ApiError(`POST /upload failed: ${res.status}`, res.status)
  return res.json() as Promise<UploadResponse>
}
