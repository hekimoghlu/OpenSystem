/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

#if ENABLE(WEBASSEMBLY)

#include "CachedResourceHandle.h"
#include "CachedScript.h"
#include "CachedScriptFetcher.h"
#include "SharedBuffer.h"
#include "WebAssemblyCachedScriptSourceProvider.h"
#include "WebAssemblyScriptBufferSourceProvider.h"
#include <JavaScriptCore/SourceCode.h>
#include <JavaScriptCore/SourceProvider.h>
#include <wtf/RefPtr.h>
#include <wtf/URL.h>

namespace WebCore {

class WebAssemblyScriptSourceCode {
public:
    WebAssemblyScriptSourceCode(CachedScript* cachedScript, Ref<CachedScriptFetcher>&& scriptFetcher)
        : m_provider(WebAssemblyCachedScriptSourceProvider::create(cachedScript, WTFMove(scriptFetcher)))
        , m_code(m_provider.copyRef())
        , m_cachedScript(cachedScript)
    {
    }

    WebAssemblyScriptSourceCode(const ScriptBuffer& source, URL&& url, Ref<JSC::ScriptFetcher>&& scriptFetcher)
        : m_provider(WebAssemblyScriptBufferSourceProvider::create(source, WTFMove(url), WTFMove(scriptFetcher)))
        , m_code(m_provider.copyRef())
    {
    }

    const JSC::SourceCode& jsSourceCode() const { return m_code; }

private:
    Ref<JSC::SourceProvider> m_provider;
    JSC::SourceCode m_code;
    CachedResourceHandle<CachedScript> m_cachedScript;
};

} // namespace WebCore

#endif // ENABLE(WEBASSEMBLY)
