/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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

#include "CachedResourceHandle.h"
#include "ReferrerPolicy.h"
#include "RequestPriority.h"
#include "ResourceLoadPriority.h"
#include "ResourceLoaderOptions.h"
#include <JavaScriptCore/ScriptFetcher.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CachedScript;
class Document;

class CachedScriptFetcher : public JSC::ScriptFetcher {
public:
    virtual CachedResourceHandle<CachedScript> requestModuleScript(Document&, const URL& sourceURL, String&& integrity, std::optional<ServiceWorkersMode>) const;

    static Ref<CachedScriptFetcher> create(const AtomString& charset);

protected:
    CachedScriptFetcher(const String& nonce, ReferrerPolicy referrerPolicy, RequestPriority fetchPriority, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree)
        : m_nonce(nonce)
        , m_charset(charset)
        , m_initiatorType(initiatorType)
        , m_isInUserAgentShadowTree(isInUserAgentShadowTree)
        , m_referrerPolicy(referrerPolicy)
        , m_fetchPriority(fetchPriority)
    {
    }

    CachedScriptFetcher(const AtomString& charset)
        : m_charset(charset)
    {
    }

    CachedResourceHandle<CachedScript> requestScriptWithCache(Document&, const URL& sourceURL, const String& crossOriginMode, String&& integrity, std::optional<ResourceLoadPriority>, std::optional<ServiceWorkersMode>) const;

private:
    String m_nonce;
    AtomString m_charset;
    AtomString m_initiatorType;
    bool m_isInUserAgentShadowTree { false };
    ReferrerPolicy m_referrerPolicy { ReferrerPolicy::EmptyString };
    RequestPriority m_fetchPriority { RequestPriority::Auto };
};

} // namespace WebCore
