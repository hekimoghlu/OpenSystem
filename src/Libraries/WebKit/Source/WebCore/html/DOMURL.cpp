/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
#include "DOMURL.h"

#include "ActiveDOMObject.h"
#include "Blob.h"
#include "BlobURL.h"
#include "MemoryCache.h"
#include "PublicURLManager.h"
#include "ResourceRequest.h"
#include "ScriptExecutionContext.h"
#include "SecurityOrigin.h"
#include "URLSearchParams.h"
#include <wtf/MainThread.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

inline DOMURL::DOMURL(URL&& completeURL)
    : m_url(WTFMove(completeURL))
{
    ASSERT(m_url.isValid());
}

ExceptionOr<Ref<DOMURL>> DOMURL::create(const String& url, const URL& base)
{
    ASSERT(base.isValid() || base.isNull());
    URL completeURL { base, url };
    if (!completeURL.isValid())
        return Exception { ExceptionCode::TypeError, makeString('"', url, "\" cannot be parsed as a URL."_s) };
    return adoptRef(*new DOMURL(WTFMove(completeURL)));
}

ExceptionOr<Ref<DOMURL>> DOMURL::create(const String& url, const String& base)
{
    URL baseURL { base };
    if (!base.isNull() && !baseURL.isValid())
        return Exception { ExceptionCode::TypeError, makeString('"', url, "\" cannot be parsed as a URL against \""_s, base, "\"."_s) };
    return create(url, baseURL);
}

DOMURL::~DOMURL() = default;

static URL parseInternal(const String& url, const String& base)
{
    URL baseURL { base };
    if (!base.isNull() && !baseURL.isValid())
        return { };
    return { baseURL, url };
}

RefPtr<DOMURL> DOMURL::parse(const String& url, const String& base)
{
    auto completeURL = parseInternal(url, base);
    if (!completeURL.isValid())
        return { };
    return adoptRef(*new DOMURL(WTFMove(completeURL)));
}

bool DOMURL::canParse(const String& url, const String& base)
{
    return parseInternal(url, base).isValid();
}

ExceptionOr<void> DOMURL::setHref(const String& url)
{
    URL completeURL { url };
    if (!completeURL.isValid())
        return Exception { ExceptionCode::TypeError };
    m_url = WTFMove(completeURL);
    if (m_searchParams)
        m_searchParams->updateFromAssociatedURL();
    return { };
}

String DOMURL::createObjectURL(ScriptExecutionContext& scriptExecutionContext, Blob& blob)
{
    return createPublicURL(scriptExecutionContext, blob);
}

String DOMURL::createPublicURL(ScriptExecutionContext& scriptExecutionContext, URLRegistrable& registrable)
{
    URL publicURL = BlobURL::createPublicURL(scriptExecutionContext.securityOrigin());
    if (publicURL.isEmpty())
        return String();

    scriptExecutionContext.publicURLManager().registerURL(publicURL, registrable);

    return publicURL.string();
}

URLSearchParams& DOMURL::searchParams()
{
    if (!m_searchParams)
        m_searchParams = URLSearchParams::create(search(), this);
    return *m_searchParams;
}

void DOMURL::revokeObjectURL(ScriptExecutionContext& scriptExecutionContext, const String& urlString)
{
    URL url { urlString };
    ResourceRequest request(url);
    request.setDomainForCachePartition(scriptExecutionContext.domainForCachePartition());

    MemoryCache::removeRequestFromSessionCaches(scriptExecutionContext, request);

    scriptExecutionContext.publicURLManager().revoke(url);
}

} // namespace WebCore
