/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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

#include "CachedResourceClient.h"
#include "CachedResourceHandle.h"
#include "CachedScript.h"
#include "CachedScriptFetcher.h"
#include <JavaScriptCore/SourceProvider.h>

namespace WebCore {

class CachedScriptSourceProvider : public JSC::SourceProvider, public CachedResourceClient {
    WTF_MAKE_TZONE_ALLOCATED(CachedScriptSourceProvider);
public:
    static Ref<CachedScriptSourceProvider> create(CachedScript* cachedScript, JSC::SourceProviderSourceType sourceType, Ref<CachedScriptFetcher>&& scriptFetcher) { return adoptRef(*new CachedScriptSourceProvider(cachedScript, sourceType, WTFMove(scriptFetcher))); }

    virtual ~CachedScriptSourceProvider()
    {
        m_cachedScript->removeClient(*this);
    }

    unsigned hash() const override;
    StringView source() const override;

private:
    CachedScriptSourceProvider(CachedScript* cachedScript, JSC::SourceProviderSourceType sourceType, Ref<CachedScriptFetcher>&& scriptFetcher)
        : SourceProvider(JSC::SourceOrigin { cachedScript->response().url(), WTFMove(scriptFetcher) }, String(cachedScript->response().url().string()), cachedScript->response().isRedirected() ? String(cachedScript->url().string()) : String(), cachedScript->requiresTelemetry() ? JSC::SourceTaintedOrigin::KnownTainted : JSC::SourceTaintedOrigin::Untainted, TextPosition(), sourceType)
        , m_cachedScript(cachedScript)
    {
        m_cachedScript->addClient(*this);
    }

    CachedResourceHandle<CachedScript> m_cachedScript;
};

inline unsigned CachedScriptSourceProvider::hash() const
{
    // Modules should always be decoded as UTF-8.
    // https://html.spec.whatwg.org/multipage/webappapis.html#fetch-a-single-module-script
    if (sourceType() == JSC::SourceProviderSourceType::Module)
        return m_cachedScript->scriptHash(CachedScript::ShouldDecodeAsUTF8Only::Yes);
    return m_cachedScript->scriptHash();
}

inline StringView CachedScriptSourceProvider::source() const
{
    // Modules should always be decoded as UTF-8.
    // https://html.spec.whatwg.org/multipage/webappapis.html#fetch-a-single-module-script
    if (sourceType() == JSC::SourceProviderSourceType::Module)
        return m_cachedScript->script(CachedScript::ShouldDecodeAsUTF8Only::Yes);
    return m_cachedScript->script();
}

} // namespace WebCore
