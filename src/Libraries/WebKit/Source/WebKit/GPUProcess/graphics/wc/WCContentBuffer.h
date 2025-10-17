/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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

#if USE(GRAPHICS_LAYER_WC)

#include "WCContentBufferIdentifier.h"
#include "WCContentBufferManager.h"
#include <WebCore/ProcessIdentifier.h>
#include <WebCore/TextureMapperPlatformLayer.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

class WCContentBuffer final : WebCore::TextureMapperPlatformLayer::Client {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WCContentBuffer);
public:
    class Client {
    public:
        virtual void platformLayerWillBeDestroyed() = 0;
    };
    
    WCContentBuffer(WCContentBufferManager& manager, WebCore::ProcessIdentifier processIdentifier, WebCore::TextureMapperPlatformLayer* platformLayer)
        : m_manager(manager)
        , m_processIdentifier(processIdentifier)
        , m_platformLayer(platformLayer)
    {
        m_platformLayer->setClient(this);
    }

    ~WCContentBuffer()
    {
        m_platformLayer->setClient(nullptr);
    }

    void setClient(Client* client)
    {
        m_client = client;
    }

    WebCore::TextureMapperPlatformLayer* platformLayer() const
    {
        return m_platformLayer;
    }

    WCContentBufferIdentifier identifier()
    {
        return m_identifier;
    }

private:
    void platformLayerWillBeDestroyed() override
    {
        if (m_client)
            m_client->platformLayerWillBeDestroyed();
        m_manager.removeContentBuffer(m_processIdentifier, *this);
    }
    void setPlatformLayerNeedsDisplay() override { }

    WCContentBufferManager& m_manager;
    WebCore::ProcessIdentifier m_processIdentifier;
    WCContentBufferIdentifier m_identifier { WCContentBufferIdentifier::generate() };
    WebCore::TextureMapperPlatformLayer* m_platformLayer;
    Client* m_client { nullptr };
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
