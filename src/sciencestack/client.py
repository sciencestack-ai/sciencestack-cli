"""ScienceStack API client."""

import httpx
import time


class ScienceStackError(Exception):
    """API error with code and message."""

    def __init__(self, code: str, message: str, status_code: int = 0):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"{code}: {message}")


class ScienceStackClient:
    """Thin wrapper around the ScienceStack REST API."""

    _RETRY_STATUS_CODES = {429, 502, 503, 504}

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://sciencestack.ai/api/v1",
        timeout: float = 30.0,
        retries: int = 0,
        retry_backoff_ms: int = 250,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff_ms = retry_backoff_ms
        self._client = httpx.Client(
            base_url=base_url,
            headers={"x-api-key": api_key},
            timeout=timeout,
            follow_redirects=True,
        )

    def _sleep_backoff(self, attempt: int, retry_after_header: str | None = None) -> None:
        if retry_after_header:
            try:
                retry_after_seconds = float(retry_after_header)
                if retry_after_seconds > 0:
                    time.sleep(retry_after_seconds)
                    return
            except ValueError:
                pass

        backoff_seconds = (self.retry_backoff_ms / 1000.0) * (2**attempt)
        time.sleep(backoff_seconds)

    def _get(self, path: str, **params) -> dict:
        """Make a GET request, return parsed JSON data."""
        # Strip None params
        params = {k: v for k, v in params.items() if v is not None}

        for attempt in range(self.retries + 1):
            try:
                resp = self._client.get(path, params=params)
            except httpx.TimeoutException as exc:
                if attempt < self.retries:
                    self._sleep_backoff(attempt)
                    continue
                raise ScienceStackError("TIMEOUT", "Request timed out", 0) from exc
            except httpx.RequestError as exc:
                if attempt < self.retries:
                    self._sleep_backoff(attempt)
                    continue
                raise ScienceStackError("NETWORK", str(exc), 0) from exc

            if resp.status_code in self._RETRY_STATUS_CODES and attempt < self.retries:
                self._sleep_backoff(attempt, resp.headers.get("retry-after"))
                continue
            break

        if not resp.content:
            raise ScienceStackError("EMPTY_RESPONSE", f"Empty response (HTTP {resp.status_code})", resp.status_code)
        try:
            data = resp.json()
        except Exception:
            raise ScienceStackError("INVALID_RESPONSE", resp.text[:200], resp.status_code)
        if resp.status_code >= 400:
            err = data.get("error", {})
            raise ScienceStackError(
                code=err.get("code", "UNKNOWN"),
                message=err.get("message", resp.text),
                status_code=resp.status_code,
            )
        return data

    def search(
        self,
        query: str,
        limit: int = 20,
        offset: int | None = None,
        field: str | None = None,
        category: str | None = None,
        sort: str = "relevance",
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict:
        return self._get(
            "/search",
            q=query,
            limit=limit,
            offset=offset,
            field=field,
            category=category,
            sort=sort,
            **{"from": from_date, "to": to_date},
        )

    def overview(self, paper_id: str) -> dict:
        return self._get(f"/papers/{paper_id}/overview")

    def batch_overview(self, paper_ids: list[str]) -> dict:
        return self._get("/papers/overview", ids=",".join(paper_ids))

    def nodes(
        self,
        paper_id: str,
        types: str | None = None,
        node_ids: str | None = None,
        format: str = "markdown",
        context: int | None = None,
        limit: int = 100,
    ) -> dict:
        return self._get(
            f"/papers/{paper_id}/nodes",
            types=types,
            nodeIds=node_ids,
            format=format,
            context=context,
            limit=limit,
        )

    def content(self, paper_id: str, format: str = "markdown") -> dict:
        return self._get(f"/papers/{paper_id}/content", format=format)

    def references(
        self,
        paper_id: str,
        cite_keys: str | None = None,
        limit: int = 100,
        offset: int | None = None,
    ) -> dict:
        return self._get(
            f"/papers/{paper_id}/references",
            citeKeys=cite_keys,
            limit=limit,
            offset=offset,
        )

    def citations(self, paper_id: str, limit: int = 100, offset: int | None = None) -> dict:
        return self._get(f"/papers/{paper_id}/citations", limit=limit, offset=offset)

    def author_papers(
        self,
        author_id: str,
        sort: str = "recent",
        limit: int = 20,
        offset: int | None = None,
    ) -> dict:
        return self._get(f"/authors/{author_id}/papers", sort=sort, limit=limit, offset=offset)

    def artifacts(
        self,
        type: str | None = None,
        field: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict:
        return self._get("/artifacts", type=type, field=field, limit=limit, offset=offset)

    def artifact(self, slug: str) -> dict:
        return self._get(f"/artifacts/{slug}")

    def close(self):
        self._client.close()
