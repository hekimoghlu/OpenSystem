/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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

#include "CachedResourceClient.h"
#include "CachedResourceHandle.h"
#include "CachedScript.h"
#include "CachedScriptFetcher.h"
#include "SharedBuffer.h"
#include <JavaScriptCore/SourceProvider.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class WebAssemblyCachedScriptSourceProvider final : public JSC::BaseWebAssemblySourceProvider, public CachedResourceClient {
    WTF_MAKE_TZONE_ALLOCATED(WebAssemblyCachedScriptSourceProvider);
public:
    static Ref<WebAssemblyCachedScriptSourceProvider> create(CachedScript* cachedScript, Ref<CachedScriptFetcher>&& scriptFetcher)
    {
        return adoptRef(*new WebAssemblyCachedScriptSourceProvider(cachedScript, JSC::SourceOrigin { cachedScript->response().url(), WTFMove(scriptFetcher) }, cachedScript->response().url().string()));
    }

    virtual ~WebAssemblyCachedScriptSourceProvider()
    {
        m_cachedScript->removeClient(*this);
    }

    unsigned hash() const final { return m_cachedScript->scriptHash(); }
    StringView source() const final { return m_cachedScript->script(); }
    size_t size() const final { return m_buffer ? m_buffer->size() : 0; }

    const uint8_t* data() final
    {
        if (!m_buffer)
            return nullptr;

        if (!m_buffer->isContiguous())
            m_buffer = m_buffer->makeContiguous();

        return downcast<SharedBuffer>(*m_buffer).span().data();
    }

    void lockUnderlyingBuffer() final
    {
        ASSERT(!m_buffer);
        m_buffer = m_cachedScript->resourceBuffer();
    }

    void unlockUnderlyingBuffer() final
    {
        ASSERT(m_buffer);
        m_buffer = nullptr;
    }

private:
    WebAssemblyCachedScriptSourceProvider(CachedScript* cachedScript, const JSC::SourceOrigin& sourceOrigin, String sourceURL)
        : BaseWebAssemblySourceProvider(sourceOrigin, WTFMove(sourceURL))
        , m_cachedScript(cachedScript)
        , m_buffer(nullptr)
    {
        m_cachedScript->addClient(*this);
    }

    CachedResourceHandle<CachedScript> m_cachedScript;
    RefPtr<FragmentedSharedBuffer> m_buffer;
};

} // namespace WebCore

#endif // ENABLE(WEBASSEMBLY)
