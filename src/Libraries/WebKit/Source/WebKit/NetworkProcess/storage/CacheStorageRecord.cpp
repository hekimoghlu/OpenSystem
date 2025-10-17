/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#include "CacheStorageRecord.h"

namespace WebKit {

CacheStorageRecordInformation::CacheStorageRecordInformation(NetworkCache::Key&& key, double insertionTime, uint64_t identifier, uint64_t updateResponseCounter, uint64_t size, URL&& url, bool hasVaryStar, HashMap<String, String>&& varyHeaders)
    : m_key(WTFMove(key))
    , m_insertionTime(insertionTime)
    , m_identifier(identifier)
    , m_updateResponseCounter(updateResponseCounter)
    , m_size(size)
    , m_url(WTFMove(url))
    , m_hasVaryStar(hasVaryStar)
    , m_varyHeaders(WTFMove(varyHeaders))
{
    RELEASE_ASSERT(!m_url.string().impl()->isAtom());
}

void CacheStorageRecordInformation::updateVaryHeaders(const WebCore::ResourceRequest& request, const WebCore::ResourceResponse::CrossThreadData& response)
{
    auto varyValue = response.httpHeaderFields.get(WebCore::HTTPHeaderName::Vary);
    if (varyValue.isNull() || response.tainting == WebCore::ResourceResponse::Tainting::Opaque || response.tainting == WebCore::ResourceResponse::Tainting::Opaqueredirect) {
        m_hasVaryStar = false;
        m_varyHeaders = { };
        return;
    }

    varyValue.split(',', [&](StringView view) {
        if (!m_hasVaryStar && view.trim(isASCIIWhitespaceWithoutFF<UChar>) == "*"_s)
            m_hasVaryStar = true;
        m_varyHeaders.add(view.toString(), request.httpHeaderField(view));
    });

    if (m_hasVaryStar)
        m_varyHeaders = { };
}

CacheStorageRecordInformation CacheStorageRecordInformation::isolatedCopy() &&
{
    return {
        crossThreadCopy(WTFMove(m_key)),
        m_insertionTime,
        m_identifier,
        m_updateResponseCounter,
        m_size,
        crossThreadCopy(WTFMove(m_url)),
        m_hasVaryStar,
        crossThreadCopy(WTFMove(m_varyHeaders))
    };
}

CacheStorageRecordInformation CacheStorageRecordInformation::isolatedCopy() const &
{
    return {
        crossThreadCopy(m_key),
        m_insertionTime,
        m_identifier,
        m_updateResponseCounter,
        m_size,
        crossThreadCopy(m_url),
        m_hasVaryStar,
        crossThreadCopy(m_varyHeaders)
    };
}

void CacheStorageRecordInformation::setURL(URL&& url)
{
    RELEASE_ASSERT(!url.string().impl()->isAtom());
    m_url = WTFMove(url);
}

} // namespace WebKit
