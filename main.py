import contextlib
import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route

mcp = FastMCP("Patient Snapshot MCP", json_response=True)

FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")
MCP_API_KEY = os.getenv("MCP_API_KEY", "change-me")
ALLOWED_ORIGINS = {
    x.strip() for x in os.getenv("ALLOWED_ORIGINS", "").split(",") if x.strip()
}

TIMEOUT = 20.0


async def fhir_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{FHIR_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Accept": "application/fhir+json"}
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def bundle_resources(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [e["resource"] for e in bundle.get("entry", []) if "resource" in e]


def human_name(patient: dict[str, Any]) -> str:
    names = patient.get("name") or []
    if not names:
        return "Unknown"
    n = names[0]
    given = " ".join(n.get("given", []))
    family = n.get("family", "")
    full = f"{given} {family}".strip()
    return full or "Unknown"


def observation_value(obs: dict[str, Any]) -> str:
    if "valueQuantity" in obs:
        q = obs["valueQuantity"]
        return f"{q.get('value', '')} {q.get('unit') or q.get('code') or ''}".strip()
    if "valueString" in obs:
        return obs["valueString"]
    if "valueCodeableConcept" in obs:
        cc = obs["valueCodeableConcept"]
        if cc.get("text"):
            return cc["text"]
        coding = cc.get("coding") or []
        if coding:
            return coding[0].get("display") or coding[0].get("code") or "Unknown"
    return "No simple value"


def observation_code(obs: dict[str, Any]) -> str:
    code = obs.get("code") or {}
    if code.get("text"):
        return code["text"]
    coding = code.get("coding") or []
    if coding:
        return coding[0].get("display") or coding[0].get("code") or "Unknown"
    return "Unknown"


@mcp.tool()
async def search_patients(name: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search patients by name and return candidate patient IDs."""
    limit = max(1, min(limit, 20))
    bundle = await fhir_get("Patient", params={"name": name, "_count": limit})
    out = []
    for p in bundle_resources(bundle):
        out.append(
            {
                "patient_id": p.get("id"),
                "name": human_name(p),
                "gender": p.get("gender"),
                "birthDate": p.get("birthDate"),
            }
        )
    return out


@mcp.tool()
async def get_patient_demographics(patient_id: str) -> dict[str, Any]:
    """Get a patient's basic demographics from FHIR."""
    p = await fhir_get(f"Patient/{patient_id}")
    return {
        "patient_id": p.get("id"),
        "name": human_name(p),
        "gender": p.get("gender"),
        "birthDate": p.get("birthDate"),
    }


@mcp.tool()
async def get_recent_observations(patient_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Get recent observations for a patient."""
    limit = max(1, min(limit, 20))
    bundle = await fhir_get(
        "Observation",
        params={"patient": patient_id, "_count": limit},
    )

    out = []
    for obs in bundle_resources(bundle):
        out.append(
            {
                "observation_id": obs.get("id"),
                "code": observation_code(obs),
                "value": observation_value(obs),
                "effective": obs.get("effectiveDateTime") or obs.get("issued") or "Unknown",
                "status": obs.get("status"),
            }
        )
    return out[:limit]


@mcp.tool()
async def build_patient_snapshot(patient_id: str) -> dict[str, Any]:
    """Build a concise read-only patient snapshot."""
    patient = await get_patient_demographics(patient_id)
    observations = await get_recent_observations(patient_id, limit=5)

    summary_lines = [
        f"Patient: {patient['name']} (ID: {patient['patient_id']})",
        f"Gender: {patient.get('gender', 'Unknown')}",
        f"Birth date: {patient.get('birthDate', 'Unknown')}",
        "",
        "Recent observations:",
    ]

    if observations:
        for obs in observations:
            summary_lines.append(
                f"- {obs['effective']}: {obs['code']} = {obs['value']} (status={obs['status']})"
            )
    else:
        summary_lines.append("- No recent observations found")

    summary_lines.append("")
    summary_lines.append(
        "Safety note: Read-only contextual summary from FHIR data. "
        "Not a diagnosis or treatment recommendation."
    )

    return {
        "patient": patient,
        "recent_observations": observations,
        "summary_text": "\n".join(summary_lines),
    }


class MCPAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/mcp"):
            origin = request.headers.get("origin")
            if origin and ALLOWED_ORIGINS and origin not in ALLOWED_ORIGINS:
                return JSONResponse({"error": "Origin not allowed"}, status_code=403)

            api_key = request.headers.get("X-API-Key")
            if api_key != MCP_API_KEY:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return await call_next(request)


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with mcp.session_manager.run():
        yield


async def health(_: Request):
    return PlainTextResponse("ok")


app = Starlette(
    routes=[
        Route("/", endpoint=health),
        Route("/healthz", endpoint=health),
        Mount("/", app=mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)

app.add_middleware(MCPAuthMiddleware)
