"""
Tests for Tavily Search API integration.
"""

import os
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.tavily.search.transformation import TavilySearchConfig
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


class TestTavilySearch:
    """
    Tests for Tavily Search functionality with mocked network responses.
    """

    @pytest.mark.asyncio
    async def test_tavily_search_request_payload(self):
        """
        Test that validates the Tavily search request payload structure without making real API calls.
        """
        # Set environment variable for API key
        os.environ["TAVILY_API_KEY"] = "test-api-key"

        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "This is a test snippet for result 1",
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "This is a test snippet for result 2",
                },
            ]
        }

        # Mock the httpx AsyncClient post method
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = mock_response

            # Make the search call
            response = await litellm.asearch(
                query="latest developments in AI",
                search_provider="tavily",
                max_results=5,
            )

            # Verify the post method was called once
            assert mock_post.call_count == 1

            # Get the actual call arguments
            call_args = mock_post.call_args

            # Verify URL
            assert call_args.kwargs["url"] == "https://api.tavily.com/search"

            # Verify headers contain Authorization
            headers = call_args.kwargs.get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-api-key"
            assert headers["Content-Type"] == "application/json"

            # Verify request payload
            json_data = call_args.kwargs.get("json")
            assert json_data is not None
            assert json_data["query"] == "latest developments in AI"
            assert json_data["max_results"] == 5

            # Verify response structure
            assert hasattr(response, "results")
            assert hasattr(response, "object")
            assert response.object == "search"
            assert len(response.results) == 2

            # Verify first result
            first_result = response.results[0]
            assert first_result.title == "Test Result 1"
            assert first_result.url == "https://example.com/1"
            assert first_result.snippet == "This is a test snippet for result 1"


def _transform(payload: object) -> SearchResponse:
    return TavilySearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload), logging_obj=None
    )


def test_tavily_results_keep_provider_order_and_return_tavily_fields_and_envelope():
    response = _transform(
        {
            "query": "litellm",
            "answer": "LiteLLM is a gateway.",
            "follow_up_questions": None,
            "images": [{"url": "https://example.com/image.png", "description": "logo"}],
            "results": [
                {
                    "title": "LiteLLM",
                    "url": "https://example.com/litellm",
                    "content": "Call every LLM API",
                    "score": 0.81,
                    "raw_content": None,
                    "favicon": "https://example.com/favicon.ico",
                    "published_date": "2026-01-15",
                },
                {"title": "Docs", "url": "https://example.com/docs", "content": "Docs", "raw_content": "Raw docs"},
            ],
            "response_time": 1.67,
            "usage": {"credits": 1},
            "request_id": "123e4567-e89b-12d3-a456-426614174111",
        }
    )

    assert response.object == "search"
    assert response.results == [
        SearchResult.model_validate(
            {
                "title": "LiteLLM",
                "url": "https://example.com/litellm",
                "snippet": "Call every LLM API",
                "date": "2026-01-15",
                "score": 0.81,
                "raw_content": None,
                "favicon": "https://example.com/favicon.ico",
            }
        ),
        SearchResult.model_validate(
            {"title": "Docs", "url": "https://example.com/docs", "snippet": "Docs", "raw_content": "Raw docs"}
        ),
    ]
    assert response.model_extra == {
        "provider_metadata": {
            "query": "litellm",
            "answer": "LiteLLM is a gateway.",
            "follow_up_questions": None,
            "images": [{"url": "https://example.com/image.png", "description": "logo"}],
            "response_time": 1.67,
            "usage": {"credits": 1},
            "request_id": "123e4567-e89b-12d3-a456-426614174111",
        }
    }


def test_tavily_result_without_title_url_or_content_gets_empty_strings():
    assert _transform({"results": [{"score": 0.5, "raw_content": None}]}).results == [
        SearchResult.model_validate(
            {
                "title": "",
                "url": "",
                "snippet": "",
                "date": None,
                "last_updated": None,
                "score": 0.5,
                "raw_content": None,
            }
        )
    ]


@pytest.mark.parametrize(
    ("tavily_fields", "expected_extras", "expected_date"),
    [
        ({}, {}, None),
        ({"raw_content": "Full page"}, {"raw_content": "Full page"}, None),
        ({"raw_content": None}, {"raw_content": None}, None),
        ({"score": 0, "favicon": ""}, {"score": 0, "favicon": ""}, None),
        (
            {"images": [{"url": "https://example.com/a.png", "description": "a"}]},
            {"images": [{"url": "https://example.com/a.png", "description": "a"}]},
            None,
        ),
        ({"a_field_tavily_adds_later": {"x": 1}}, {"a_field_tavily_adds_later": {"x": 1}}, None),
        ({"published_date": "2026-01-15"}, {}, "2026-01-15"),
        ({"published_date": None}, {}, None),
        ({"snippet": "from tavily", "date": "x", "last_updated": "y"}, {}, None),
    ],
)
def test_tavily_result_attaches_every_unmapped_field_tavily_returned(
    tavily_fields: dict[str, object], expected_extras: dict[str, object], expected_date: str | None
):
    response: Final = _transform(
        {
            "results": [
                {"title": "Plain", "url": "https://example.com/plain", "content": "plain"},
                {"title": "Title", "url": "https://example.com", "content": "Snippet", **tavily_fields},
            ]
        }
    )

    assert [(result.model_extra, result.date, result.snippet) for result in response.results] == [
        ({}, None, "plain"),
        (expected_extras, expected_date, "Snippet"),
    ]


@pytest.mark.parametrize(
    ("payload", "expected_metadata"),
    [
        ({}, {}),
        ({"results": []}, {}),
        ({"request_id": "abc"}, {"request_id": "abc"}),
        (
            {"answer": "a", "usage": {"credits": 2}, "auto_parameters": {"topic": "general"}, "results": [{}]},
            {"answer": "a", "usage": {"credits": 2}, "auto_parameters": {"topic": "general"}},
        ),
    ],
)
def test_tavily_response_keeps_the_tavily_envelope_as_provider_metadata(
    payload: dict[str, object], expected_metadata: dict[str, object]
):
    response: Final = _transform(payload)

    assert response.model_extra == {"provider_metadata": expected_metadata}
    assert response.model_dump(mode="json")["provider_metadata"] == expected_metadata


@pytest.mark.parametrize(
    "payload",
    [{}, {"query": "litellm", "answer": None}, {"results": []}, {"results": ""}, {"results": {}}],
)
def test_tavily_response_without_results_is_empty(payload):
    assert _transform(payload).results == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", None),
        ("title", 7),
        ("url", None),
        ("url", ["https://example.com"]),
        ("content", None),
        ("content", {"text": "snippet"}),
        ("published_date", 7),
    ],
)
def test_tavily_result_with_field_of_wrong_type_is_rejected(field, value):
    result = {"title": "T", "url": "https://example.com", "content": "S", field: value}

    with pytest.raises(ValidationError):
        _transform({"results": [result]})


def test_tavily_result_with_several_fields_of_wrong_type_reports_every_field():
    result = {"title": 7, "url": None, "content": ["snippet"]}

    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": [result]})

    assert [error["loc"] for error in exc_info.value.errors()] == [("title",), ("url",), ("snippet",)]


def test_tavily_malformed_result_is_rejected_without_naming_its_position():
    results = [{}] * 429 + ["not-an-object"]

    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": results})

    assert "429" not in str(exc_info.value)


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.mark.parametrize(
    "api_base",
    [
        "https://gateway.example.com/tavily",
        "https://gateway.example.com/tavily/",
        "https://gateway.example.com/tavily/search/",
    ],
)
def test_get_complete_url_ignores_a_trailing_slash_on_api_base(api_base: str) -> None:
    assert (
        TavilySearchConfig().get_complete_url(api_base=api_base, optional_params={})
        == "https://gateway.example.com/tavily/search"
    )
