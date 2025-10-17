/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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
#include "CachedScript.h"
#include "CachedScriptFetcher.h"
#include "CachedScriptSourceProvider.h"
#include "ScriptBufferSourceProvider.h"
#include <JavaScriptCore/SourceCode.h>
#include <JavaScriptCore/SourceProvider.h>
#include <wtf/RefPtr.h>
#include <wtf/URL.h>
#include <wtf/text/TextPosition.h>

namespace WebCore {

class ScriptSourceCode {
public:
    ScriptSourceCode(const String& source, JSC::SourceTaintedOrigin sourceTaintedOrigin, URL&& url = URL(), const TextPosition& startPosition = TextPosition(), JSC::SourceProviderSourceType sourceType = JSC::SourceProviderSourceType::Program)
        : m_provider(JSC::StringSourceProvider::create(source, JSC::SourceOrigin { url }, url.string(), sourceTaintedOrigin, startPosition, sourceType))
        , m_code(m_provider.copyRef(), startPosition.m_line.oneBasedInt(), startPosition.m_column.oneBasedInt())
    {
    }

    ScriptSourceCode(const ScriptBuffer& source, URL&& url = URL(), URL&& preRedirectURL = URL(), const TextPosition& startPosition = TextPosition(), JSC::SourceProviderSourceType sourceType = JSC::SourceProviderSourceType::Program)
        : m_provider(ScriptBufferSourceProvider::create(source, JSC::SourceOrigin { url }, url.string(), preRedirectURL.string(), startPosition, sourceType))
        , m_code(m_provider.copyRef(), startPosition.m_line.oneBasedInt(), startPosition.m_column.oneBasedInt())
    {
    }

    ScriptSourceCode(CachedScript* cachedScript, JSC::SourceProviderSourceType sourceType, Ref<CachedScriptFetcher>&& scriptFetcher)
        : m_provider(CachedScriptSourceProvider::create(cachedScript, sourceType, WTFMove(scriptFetcher)))
        , m_code(m_provider.copyRef())
        , m_cachedScript(cachedScript)
    {
    }

    ScriptSourceCode(const String& source, JSC::SourceTaintedOrigin sourceTaintedOrigin, URL&& url, const TextPosition& startPosition, JSC::SourceProviderSourceType sourceType, Ref<JSC::ScriptFetcher>&& scriptFetcher)
        : m_provider(JSC::StringSourceProvider::create(source, JSC::SourceOrigin { url, WTFMove(scriptFetcher) }, url.string(), sourceTaintedOrigin, startPosition, sourceType))
        , m_code(m_provider.copyRef(), startPosition.m_line.oneBasedInt(), startPosition.m_column.oneBasedInt())
    {
    }

    ScriptSourceCode(const ScriptBuffer& source, URL&& url, URL&& preRedirectURL, const TextPosition& startPosition, JSC::SourceProviderSourceType sourceType, Ref<JSC::ScriptFetcher>&& scriptFetcher)
        : m_provider(ScriptBufferSourceProvider::create(source, JSC::SourceOrigin { url, WTFMove(scriptFetcher) }, url.string(), preRedirectURL.string(), startPosition, sourceType))
        , m_code(m_provider.copyRef(), startPosition.m_line.oneBasedInt(), startPosition.m_column.oneBasedInt())
    {
    }

    bool isEmpty() const { return m_code.length() == 0; }

    const JSC::SourceCode& jsSourceCode() const { return m_code; }

    JSC::SourceProvider& provider() { return m_provider.get(); }
    StringView source() const { return m_provider->source(); }

    int startLine() const { return m_code.firstLine().oneBasedInt(); }
    int startColumn() const { return m_code.startColumn().oneBasedInt(); }

    CachedScript* cachedScript() const { return m_cachedScript.get(); }

    const URL& url() const { return m_provider->sourceOrigin().url(); }
    
private:
    Ref<JSC::SourceProvider> m_provider;
    JSC::SourceCode m_code;
    CachedResourceHandle<CachedScript> m_cachedScript;
};

} // namespace WebCore
