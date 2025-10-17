/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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

#include "NetworkCacheKey.h"
#include <WebCore/DOMCacheEngine.h>
#include <WebCore/HTTPParsers.h>
#include <WebCore/ResourceResponse.h>

namespace WebKit {

class CacheStorageRecordInformation {
public:
    CacheStorageRecordInformation() = default;
    CacheStorageRecordInformation(NetworkCache::Key&&, double insertionTime, uint64_t identifier, uint64_t updateResponseCounter, uint64_t size, URL&&, bool hasVaryStar, HashMap<String, String>&& varyHeaders);
    void updateVaryHeaders(const WebCore::ResourceRequest&, const WebCore::ResourceResponse::CrossThreadData&);
    CacheStorageRecordInformation isolatedCopy() &&;
    CacheStorageRecordInformation isolatedCopy() const &;

    const NetworkCache::Key& key() const { return m_key; }
    double insertionTime() const { return m_insertionTime; }
    uint64_t identifier() const { return m_identifier; }
    uint64_t updateResponseCounter() const { return m_updateResponseCounter; }
    uint64_t size() const { return m_size; }
    const URL& url() const { return m_url; }
    bool hasVaryStar() const { return m_hasVaryStar; }
    const HashMap<String, String>& varyHeaders() const { return m_varyHeaders; }

    void setKey(NetworkCache::Key&& key) { m_key = WTFMove(key); }
    void setSize(uint64_t size) { m_size = size; }
    void setIdentifier(uint64_t identifier) { m_identifier = identifier; }
    void setUpdateResponseCounter(double updateResponseCounter) { m_updateResponseCounter = updateResponseCounter; }
    void setInsertionTime(double insertionTime) { m_insertionTime = insertionTime; }
    void setURL(URL&&);

private:
    NetworkCache::Key m_key;
    double m_insertionTime { 0 };
    uint64_t m_identifier { 0 };
    uint64_t m_updateResponseCounter { 0 };
    uint64_t m_size { 0 };
    URL m_url;
    bool m_hasVaryStar { false };
    HashMap<String, String> m_varyHeaders;
};

struct CacheStorageRecord {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    CacheStorageRecord(const CacheStorageRecord&) = delete;
    CacheStorageRecord& operator=(const CacheStorageRecord&) = delete;
    CacheStorageRecord() = default;
    CacheStorageRecord(CacheStorageRecord&&) = default;
    CacheStorageRecord& operator=(CacheStorageRecord&&) = default;
    CacheStorageRecord(const CacheStorageRecordInformation& info, WebCore::FetchHeaders::Guard requestHeadersGuard, const WebCore::ResourceRequest& request, WebCore::FetchOptions options, const String& referrer, WebCore::FetchHeaders::Guard responseHeadersGuard, WebCore::ResourceResponse::CrossThreadData&& responseData, uint64_t responseBodySize, WebCore::DOMCacheEngine::ResponseBody&& responseBody)
        : info(info)
        , requestHeadersGuard(requestHeadersGuard)
        , request(request)
        , options(options)
        , referrer(referrer)
        , responseHeadersGuard(responseHeadersGuard)
        , responseData(WTFMove(responseData))
        , responseBodySize(responseBodySize)
        , responseBody(WTFMove(responseBody))
    {
    }

    CacheStorageRecord isolatedCopy() && {
        return {
            crossThreadCopy(WTFMove(info)),
            requestHeadersGuard,
            crossThreadCopy(WTFMove(request)),
            crossThreadCopy(WTFMove(options)),
            crossThreadCopy(WTFMove(referrer)),
            responseHeadersGuard,
            crossThreadCopy(WTFMove(responseData)),
            responseBodySize,
            WebCore::DOMCacheEngine::isolatedResponseBody(WTFMove(responseBody))
        };
    }

    CacheStorageRecordInformation info;
    WebCore::FetchHeaders::Guard requestHeadersGuard;
    WebCore::ResourceRequest request;
    WebCore::FetchOptions options;
    String referrer;
    WebCore::FetchHeaders::Guard responseHeadersGuard;
    WebCore::ResourceResponse::CrossThreadData responseData;
    uint64_t responseBodySize;
    WebCore::DOMCacheEngine::ResponseBody responseBody;
};

}
