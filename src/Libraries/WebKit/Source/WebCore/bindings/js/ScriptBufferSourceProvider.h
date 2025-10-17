/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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

#include "ScriptBuffer.h"
#include <JavaScriptCore/SourceProvider.h>

namespace WebCore {
class AbstractScriptBufferHolder;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::AbstractScriptBufferHolder> : std::true_type { };
}

namespace WebCore {

class AbstractScriptBufferHolder : public CanMakeWeakPtr<AbstractScriptBufferHolder> {
public:
    virtual void clearDecodedData() = 0;
    virtual void tryReplaceScriptBuffer(const ScriptBuffer&) = 0;

    virtual ~AbstractScriptBufferHolder() { }
};

class ScriptBufferSourceProvider final : public JSC::SourceProvider, public AbstractScriptBufferHolder {
    WTF_MAKE_TZONE_ALLOCATED(ScriptBufferSourceProvider);
public:
    static Ref<ScriptBufferSourceProvider> create(const ScriptBuffer& scriptBuffer, const JSC::SourceOrigin& sourceOrigin, String sourceURL, String preRedirectURL, const TextPosition& startPosition = TextPosition(), JSC::SourceProviderSourceType sourceType = JSC::SourceProviderSourceType::Program)
    {
        return adoptRef(*new ScriptBufferSourceProvider(scriptBuffer, sourceOrigin, WTFMove(sourceURL), WTFMove(preRedirectURL), startPosition, sourceType));
    }

    unsigned hash() const final
    {
        if (!m_scriptHash)
            source();
        return m_scriptHash;
    }

    StringView source() const final
    {
        if (m_scriptBuffer.isEmpty())
            return emptyString();

        if (!m_contiguousBuffer && (!m_containsOnlyASCII || *m_containsOnlyASCII))
            m_contiguousBuffer = m_scriptBuffer.buffer()->makeContiguous();
        if (!m_containsOnlyASCII) {
            m_containsOnlyASCII = charactersAreAllASCII(m_contiguousBuffer->span());
            if (*m_containsOnlyASCII)
                m_scriptHash = StringHasher::computeHashAndMaskTop8Bits(m_contiguousBuffer->span());
        }
        if (*m_containsOnlyASCII)
            return m_contiguousBuffer->span();

        if (!m_cachedScriptString) {
            m_cachedScriptString = m_scriptBuffer.toString();
            if (!m_scriptHash)
                m_scriptHash = m_cachedScriptString.impl()->hash();
        }

        return m_cachedScriptString;
    }

    void clearDecodedData() final
    {
        m_cachedScriptString = String();
    }

    void tryReplaceScriptBuffer(const ScriptBuffer& scriptBuffer) final
    {
        // If this new file-mapped script buffer is identical to the one we have, then replace
        // ours to save dirty memory.
        if (m_scriptBuffer != scriptBuffer)
            return;

        m_scriptBuffer = scriptBuffer;
        m_contiguousBuffer = nullptr;
    }

private:
    ScriptBufferSourceProvider(const ScriptBuffer& scriptBuffer, const JSC::SourceOrigin& sourceOrigin, String&& sourceURL, String&& preRedirectURL, const TextPosition& startPosition, JSC::SourceProviderSourceType sourceType)
        : JSC::SourceProvider(sourceOrigin, WTFMove(sourceURL), WTFMove(preRedirectURL), JSC::SourceTaintedOrigin::Untainted, startPosition, sourceType)
        , m_scriptBuffer(scriptBuffer)
    {
    }

    ScriptBuffer m_scriptBuffer;
    mutable RefPtr<SharedBuffer> m_contiguousBuffer;
    mutable unsigned m_scriptHash { 0 };
    mutable String m_cachedScriptString;
    mutable std::optional<bool> m_containsOnlyASCII;
};

} // namespace WebCore
