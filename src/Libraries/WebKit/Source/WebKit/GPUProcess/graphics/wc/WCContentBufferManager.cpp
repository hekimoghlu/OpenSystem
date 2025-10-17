/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#include "WCContentBufferManager.h"

#if USE(GRAPHICS_LAYER_WC)

#include "WCContentBuffer.h"
#include <wtf/HashSet.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

class WCContentBufferManager::ProcessInfo {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WCContentBufferManager);
public:
    ProcessInfo(WCContentBufferManager& manager, WebCore::ProcessIdentifier processIdentifier)
        : m_manager(manager)
        , m_processIdentifier(processIdentifier) { }

    std::optional<WCContentBufferIdentifier> acquireContentBufferIdentifier(WebCore::TextureMapperPlatformLayer* platformLayer)
    {
        // FIXME: TextureMapperGCGLPlatformLayer doesn't support double buffering yet.
        // TextureMapperPlatformLayer can acquire a single WCContentBufferIdentifier.
        auto contentBufferAddResult = m_contentBuffers.ensure(platformLayer, [&] {
            return makeUnique<WCContentBuffer>(m_manager, m_processIdentifier, platformLayer);
        });
        WCContentBuffer* buffer = contentBufferAddResult.iterator->value.get();
        auto identifier = buffer->identifier();
        auto identifierAddResult = m_validIdentifiers.add(identifier, buffer);
        if (!identifierAddResult.isNewEntry)
            return std::nullopt;
        return identifier;
    }

    WCContentBuffer* releaseContentBufferIdentifier(WCContentBufferIdentifier identifier)
    {
        return m_validIdentifiers.take(identifier);
    }

    void removeContentBuffer(WCContentBuffer& contentBuffer)
    {
        m_validIdentifiers.remove(contentBuffer.identifier());
        m_contentBuffers.remove(contentBuffer.platformLayer());
    }

private:
    WCContentBufferManager& m_manager;
    WebCore::ProcessIdentifier m_processIdentifier;
    HashMap<WebCore::TextureMapperPlatformLayer*, std::unique_ptr<WCContentBuffer>> m_contentBuffers;
    HashMap<WCContentBufferIdentifier, WCContentBuffer*> m_validIdentifiers;
};

WCContentBufferManager& WCContentBufferManager::singleton()
{
    static NeverDestroyed<WCContentBufferManager> contentBufferManager;
    return contentBufferManager;
}

std::optional<WCContentBufferIdentifier> WCContentBufferManager::acquireContentBufferIdentifier(WebCore::ProcessIdentifier processIdentifier, WebCore::TextureMapperPlatformLayer* platformLayer)
{
    auto processAddResult = m_processMap.ensure(processIdentifier, [&] {
        return makeUnique<ProcessInfo>(*this, processIdentifier);
    });
    return processAddResult.iterator->value->acquireContentBufferIdentifier(platformLayer);
}

WCContentBuffer* WCContentBufferManager::releaseContentBufferIdentifier(WebCore::ProcessIdentifier processIdentifier, WCContentBufferIdentifier contentBufferIdentifier)
{
    ASSERT(m_processMap.contains(processIdentifier));
    return m_processMap.get(processIdentifier)->releaseContentBufferIdentifier(contentBufferIdentifier);
}

void WCContentBufferManager::removeContentBuffer(WebCore::ProcessIdentifier processIdentifier, WCContentBuffer& contentBuffer)
{
    ASSERT(m_processMap.contains(processIdentifier));
    m_processMap.get(processIdentifier)->removeContentBuffer(contentBuffer);
}

void WCContentBufferManager::removeAllContentBuffersForProcess(WebCore::ProcessIdentifier processIdentifier)
{
    m_processMap.remove(processIdentifier);
}

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
