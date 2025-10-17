/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "config.h"
#include "DOMCacheEngine.h"

#include "CacheQueryOptions.h"
#include "CrossOriginAccessControl.h"
#include "Exception.h"
#include "HTTPParsers.h"
#include "ScriptExecutionContext.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

namespace DOMCacheEngine {

Exception convertToException(Error error)
{
    switch (error) {
    case Error::NotImplemented:
        return Exception { ExceptionCode::NotSupportedError, "Not implemented"_s };
    case Error::ReadDisk:
        return Exception { ExceptionCode::TypeError, "Failed reading data from the file system"_s };
    case Error::WriteDisk:
        return Exception { ExceptionCode::TypeError, "Failed writing data to the file system"_s };
    case Error::QuotaExceeded:
        return Exception { ExceptionCode::QuotaExceededError, "Quota exceeded"_s };
    case Error::Internal:
        return Exception { ExceptionCode::TypeError, "Internal error"_s };
    case Error::Stopped:
        return Exception { ExceptionCode::TypeError, "Context is stopped"_s };
    case Error::CORP:
        return Exception { ExceptionCode::TypeError, "Cross-Origin-Resource-Policy failure"_s };
    }
    ASSERT_NOT_REACHED();
    return Exception { ExceptionCode::TypeError, "Connection stopped"_s };
}

Exception convertToExceptionAndLog(ScriptExecutionContext* context, Error error)
{
    auto exception = convertToException(error);
    if (context)
        context->addConsoleMessage(MessageSource::JS, MessageLevel::Error, makeString("Cache API operation failed: "_s, exception.message()));
    return exception;
}

static inline bool matchURLs(const ResourceRequest& request, const URL& cachedURL, const CacheQueryOptions& options)
{
    ASSERT(options.ignoreMethod || request.httpMethod() == "GET"_s);

    URL requestURL = request.url();
    URL cachedRequestURL = cachedURL;

    if (options.ignoreSearch) {
        requestURL.setQuery({ });
        cachedRequestURL.setQuery({ });
    }
    return equalIgnoringFragmentIdentifier(requestURL, cachedRequestURL);
}

bool queryCacheMatch(const ResourceRequest& request, const ResourceRequest& cachedRequest, const ResourceResponse& cachedResponse, const CacheQueryOptions& options)
{
    if (!matchURLs(request, cachedRequest.url(), options))
        return false;

    if (options.ignoreVary)
        return true;

    String varyValue = cachedResponse.httpHeaderField(WebCore::HTTPHeaderName::Vary);
    if (varyValue.isNull())
        return true;

    bool isVarying = false;
    varyValue.split(',', [&](StringView view) {
        if (isVarying)
            return;
        auto nameView = view.trim(isASCIIWhitespaceWithoutFF<UChar>);
        if (nameView == "*"_s) {
            isVarying = true;
            return;
        }
        isVarying = cachedRequest.httpHeaderField(nameView) != request.httpHeaderField(nameView);
    });

    return !isVarying;
}

bool queryCacheMatch(const ResourceRequest& request, const URL& url, bool hasVaryStar, const HashMap<String, String>& varyHeaders, const CacheQueryOptions& options)
{
    if (!matchURLs(request, url, options))
        return false;

    if (options.ignoreVary)
        return true;

    if (hasVaryStar)
        return false;

    for (const auto& pair : varyHeaders) {
        if (pair.value != request.httpHeaderField(pair.key))
            return false;
    }
    return true;
}

ResponseBody isolatedResponseBody(const ResponseBody& body)
{
    return WTF::switchOn(body, [](const Ref<FormData>& formData) -> ResponseBody {
        return formData->isolatedCopy();
    }, [](const Ref<SharedBuffer>& buffer) -> ResponseBody {
        return buffer.copyRef(); // SharedBuffer are immutable and can be returned as-is.
    }, [](const std::nullptr_t&) -> ResponseBody {
        return DOMCacheEngine::ResponseBody { };
    });
}

ResponseBody copyResponseBody(const ResponseBody& body)
{
    return WTF::switchOn(body, [](const Ref<FormData>& formData) -> ResponseBody {
        return formData.copyRef();
    }, [](const Ref<SharedBuffer>& buffer) -> ResponseBody {
        return buffer.copyRef();
    }, [](const std::nullptr_t&) -> ResponseBody {
        return DOMCacheEngine::ResponseBody { };
    });
}

Record Record::copy() const
{
    return Record { identifier, updateResponseCounter, requestHeadersGuard, request, options, referrer, responseHeadersGuard, response, copyResponseBody(responseBody), responseBodySize };
}

CrossThreadRecord toCrossThreadRecord(Record&& record)
{
    return CrossThreadRecord {
        record.identifier,
        record.updateResponseCounter,
        record.requestHeadersGuard,
        WTFMove(record.request).isolatedCopy(),
        WTFMove(record.options).isolatedCopy(),
        WTFMove(record.referrer).isolatedCopy(),
        record.responseHeadersGuard,
        record.response.crossThreadData(),
        isolatedResponseBody(record.responseBody),
        record.responseBodySize
    };
}

Record fromCrossThreadRecord(CrossThreadRecord&& record)
{
    return Record {
        record.identifier,
        record.updateResponseCounter,
        record.requestHeadersGuard,
        WTFMove(record.request),
        WTFMove(record.options),
        WTFMove(record.referrer),
        record.responseHeadersGuard,
        ResourceResponse::fromCrossThreadData(WTFMove(record.response)),
        WTFMove(record.responseBody),
        record.responseBodySize
    };
}

CrossThreadRecord CrossThreadRecord::isolatedCopy() &&
{
    return CrossThreadRecord {
        identifier,
        updateResponseCounter,
        requestHeadersGuard,
        WTFMove(request).isolatedCopy(),
        WTFMove(options).isolatedCopy(),
        WTFMove(referrer).isolatedCopy(),
        responseHeadersGuard,
        WTFMove(response).isolatedCopy(),
        isolatedResponseBody(responseBody),
        responseBodySize
    };
}

} // namespace DOMCacheEngine

} // namespace WebCore

