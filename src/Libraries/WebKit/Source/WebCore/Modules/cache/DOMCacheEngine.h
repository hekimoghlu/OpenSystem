/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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
#pragma once

#include "DOMCacheIdentifier.h"
#include "FetchHeaders.h"
#include "FetchOptions.h"
#include "ResourceRequest.h"
#include "ResourceResponse.h"
#include "SharedBuffer.h"
#include <wtf/CompletionHandler.h>

namespace WebCore {

class ScriptExecutionContext;

struct CacheQueryOptions;

namespace DOMCacheEngine {

enum class Error : uint8_t {
    NotImplemented,
    ReadDisk,
    WriteDisk,
    QuotaExceeded,
    Internal,
    Stopped,
    CORP
};

Exception convertToException(Error);
Exception convertToExceptionAndLog(ScriptExecutionContext*, Error);

WEBCORE_EXPORT bool queryCacheMatch(const ResourceRequest& request, const ResourceRequest& cachedRequest, const ResourceResponse&, const CacheQueryOptions&);
WEBCORE_EXPORT bool queryCacheMatch(const ResourceRequest&, const URL&, bool hasVaryStar, const HashMap<String, String>& varyHeaders, const CacheQueryOptions&);

using ResponseBody = std::variant<std::nullptr_t, Ref<FormData>, Ref<SharedBuffer>>;
WEBCORE_EXPORT ResponseBody isolatedResponseBody(const ResponseBody&);
WEBCORE_EXPORT ResponseBody copyResponseBody(const ResponseBody&);

struct Record {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    WEBCORE_EXPORT Record copy() const;

    uint64_t identifier;
    uint64_t updateResponseCounter;

    FetchHeaders::Guard requestHeadersGuard;
    ResourceRequest request;
    FetchOptions options;
    String referrer;

    FetchHeaders::Guard responseHeadersGuard;
    ResourceResponse response;
    ResponseBody responseBody;
    uint64_t responseBodySize;
};

struct CrossThreadRecord {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    CrossThreadRecord(const CrossThreadRecord&) = delete;
    CrossThreadRecord& operator=(const CrossThreadRecord&) = delete;
    CrossThreadRecord() = default;
    CrossThreadRecord(CrossThreadRecord&&) = default;
    CrossThreadRecord& operator=(CrossThreadRecord&&) = default;
    CrossThreadRecord(uint64_t identifier, uint64_t updateResponseCounter, FetchHeaders::Guard requestHeadersGuard, ResourceRequest&& request, FetchOptions options, String&& referrer, FetchHeaders::Guard responseHeadersGuard, ResourceResponse::CrossThreadData&& response, ResponseBody&& responseBody, uint64_t responseBodySize)
        : identifier(identifier)
        , updateResponseCounter(updateResponseCounter)
        , requestHeadersGuard(requestHeadersGuard)
        , request(WTFMove(request))
        , options(options)
        , referrer(WTFMove(referrer))
        , responseHeadersGuard(responseHeadersGuard)
        , response(WTFMove(response))
        , responseBody(WTFMove(responseBody))
        , responseBodySize(responseBodySize)
    {
    }
    WEBCORE_EXPORT CrossThreadRecord isolatedCopy() &&;

    uint64_t identifier;
    uint64_t updateResponseCounter;
    FetchHeaders::Guard requestHeadersGuard;
    ResourceRequest request;
    FetchOptions options;
    String referrer;
    FetchHeaders::Guard responseHeadersGuard;
    ResourceResponse::CrossThreadData response;
    ResponseBody responseBody;
    uint64_t responseBodySize;
};

WEBCORE_EXPORT CrossThreadRecord toCrossThreadRecord(Record&&);
WEBCORE_EXPORT Record fromCrossThreadRecord(CrossThreadRecord&&);

struct CacheInfo {
    DOMCacheIdentifier identifier;
    String name;

    CacheInfo isolatedCopy() const & { return { identifier, name.isolatedCopy() }; }
    CacheInfo isolatedCopy() && { return { identifier, WTFMove(name).isolatedCopy() }; }
};

struct CacheInfos {
    Vector<CacheInfo> infos;
    uint64_t updateCounter;

    CacheInfos isolatedCopy() const & { return { crossThreadCopy(infos), updateCounter }; }
    CacheInfos isolatedCopy() && { return { crossThreadCopy(WTFMove(infos)), updateCounter }; }
};

struct CacheIdentifierOperationResult {
    DOMCacheIdentifier identifier;
    // True in case storing cache list on the filesystem failed.
    bool hadStorageError { false };
};

using CacheIdentifierOrError = Expected<CacheIdentifierOperationResult, Error>;
using CacheIdentifierCallback = CompletionHandler<void(const CacheIdentifierOrError&)>;

using RemoveCacheIdentifierOrError = Expected<bool, Error>;
using RemoveCacheIdentifierCallback = CompletionHandler<void(const RemoveCacheIdentifierOrError&)>;

using RecordIdentifiersOrError = Expected<Vector<uint64_t>, Error>;
using RecordIdentifiersCallback = CompletionHandler<void(RecordIdentifiersOrError&&)>;


using CacheInfosOrError = Expected<CacheInfos, Error>;
using CacheInfosCallback = CompletionHandler<void(CacheInfosOrError&&)>;

using RecordsOrError = Expected<Vector<Record>, Error>;
using RecordsCallback = CompletionHandler<void(RecordsOrError&&)>;

using CrossThreadRecordsOrError = Expected<Vector<CrossThreadRecord>, Error>;
using CrossThreadRecordsCallback = CompletionHandler<void(CrossThreadRecordsOrError&&)>;

using CompletionCallback = CompletionHandler<void(std::optional<Error>&&)>;

} // namespace DOMCacheEngine

} // namespace WebCore
