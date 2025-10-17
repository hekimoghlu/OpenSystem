/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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

#include "ScriptBufferSourceProvider.h"
#include <JavaScriptCore/SourceProvider.h>

namespace WebCore {

class WebAssemblyScriptBufferSourceProvider final : public JSC::BaseWebAssemblySourceProvider, public AbstractScriptBufferHolder {
    WTF_MAKE_TZONE_ALLOCATED(WebAssemblyScriptBufferSourceProvider);
public:
    static Ref<WebAssemblyScriptBufferSourceProvider> create(const ScriptBuffer& scriptBuffer, URL&& sourceURL, Ref<JSC::ScriptFetcher>&& scriptFetcher)
    {
        return adoptRef(*new WebAssemblyScriptBufferSourceProvider(scriptBuffer, JSC::SourceOrigin { WTFMove(sourceURL), WTFMove(scriptFetcher) }, sourceURL.string()));
    }

    unsigned hash() const final
    {
        return m_source.impl()->hash();
    }

    StringView source() const final
    {
        return m_source;
    }

    size_t size() const final { return m_buffer ? m_buffer->size() : 0; }

    const uint8_t* data() final
    {
        if (!m_buffer)
            return nullptr;

        ASSERT(m_buffer->isContiguous());
        return downcast<SharedBuffer>(*m_buffer).span().data();
    }

    void lockUnderlyingBuffer() final
    {
        ASSERT(!m_buffer);
        m_buffer = m_scriptBuffer.buffer();

        if (!m_buffer)
            return;

        if (!m_buffer->isContiguous())
            m_buffer = m_buffer->makeContiguous();
    }

    void unlockUnderlyingBuffer() final
    {
        ASSERT(m_buffer);
        m_buffer = nullptr;
    }

    void clearDecodedData() final { }

    void tryReplaceScriptBuffer(const ScriptBuffer& scriptBuffer) final
    {
        if (m_scriptBuffer != scriptBuffer)
            return;

        m_scriptBuffer = scriptBuffer;
    }

private:
    WebAssemblyScriptBufferSourceProvider(const ScriptBuffer& scriptBuffer, const JSC::SourceOrigin& sourceOrigin, String sourceURL)
        : BaseWebAssemblySourceProvider(sourceOrigin, WTFMove(sourceURL))
        , m_scriptBuffer(scriptBuffer)
        , m_buffer(nullptr)
        , m_source("[WebAssembly source]"_s)
    {
    }

    ScriptBuffer m_scriptBuffer;
    RefPtr<const FragmentedSharedBuffer> m_buffer;
    String m_source;
};

} // namespace WebCore

#endif // ENABLE(WEBASSEMBLY)
