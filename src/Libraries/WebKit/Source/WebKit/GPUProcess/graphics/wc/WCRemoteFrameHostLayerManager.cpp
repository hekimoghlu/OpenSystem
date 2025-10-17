/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#include "WCRemoteFrameHostLayerManager.h"

#if USE(GRAPHICS_LAYER_WC)

#include <WebCore/BitmapTexture.h>
#include <WebCore/TextureMapper.h>
#include <WebCore/TextureMapperPlatformLayer.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

class WCRemoteFrameHostLayer final : public WebCore::TextureMapperPlatformLayer {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WCRemoteFrameHostLayer);
public:
    void paintToTextureMapper(WebCore::TextureMapper& textureMapper, const WebCore::FloatRect& targetRect, const WebCore::TransformationMatrix& modelViewMatrix, float opacity) override
    {
        if (m_texture)
            textureMapper.drawTexture(*m_texture, targetRect, modelViewMatrix, opacity);
    }

    void setTexture(RefPtr<WebCore::BitmapTexture> texture)
    {
        m_texture = texture;
    }

private:
    RefPtr<WebCore::BitmapTexture> m_texture;
};

class WCRemoteFrameHostLayerManager::RemoteFrameHostLayerData final {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WCRemoteFrameHostLayerManager);
public:
    RemoteFrameHostLayerData(WebCore::ProcessIdentifier webProcessIdentifier)
        : m_webProcessIdentifier(webProcessIdentifier)
    {
    }

    WebCore::ProcessIdentifier webProcessIdentifier() const { return m_webProcessIdentifier; }
    void setWebProcessIdentifier(WebCore::ProcessIdentifier webProcessIdentifier) { m_webProcessIdentifier = webProcessIdentifier; }
    WCRemoteFrameHostLayer& layer() { return m_layer; }

private:
    // A web process identifier currently owning the texture. Remote frame process or frame host process.
    WebCore::ProcessIdentifier m_webProcessIdentifier;
    WCRemoteFrameHostLayer m_layer;
};

WCRemoteFrameHostLayerManager& WCRemoteFrameHostLayerManager::singleton()
{
    static NeverDestroyed<WCRemoteFrameHostLayerManager> manager;
    return manager;
}

WebCore::TextureMapperPlatformLayer* WCRemoteFrameHostLayerManager::acquireRemoteFrameHostLayer(WebCore::LayerHostingContextIdentifier layerHostingContextIdentifier, WebCore::ProcessIdentifier webProcessIdentifier)
{
    auto& data = *m_layers.ensure(layerHostingContextIdentifier, [&] {
        return makeUnique<RemoteFrameHostLayerData>(webProcessIdentifier);
    }).iterator->value;
    // Transfer the data ownership to the frame host process.
    data.setWebProcessIdentifier(webProcessIdentifier);
    return &data.layer();
}

void WCRemoteFrameHostLayerManager::releaseRemoteFrameHostLayer(WebCore::LayerHostingContextIdentifier layerHostingContextIdentifier)
{
    m_layers.remove(layerHostingContextIdentifier);
}

void WCRemoteFrameHostLayerManager::updateTexture(WebCore::LayerHostingContextIdentifier layerHostingContextIdentifier, WebCore::ProcessIdentifier webProcessIdentifier, RefPtr<WebCore::BitmapTexture> texture)
{
    m_layers.ensure(layerHostingContextIdentifier, [&] {
        // Create a new data if the frame host didn't create it yet. The initial owner is the remote frame process.
        return makeUnique<RemoteFrameHostLayerData>(webProcessIdentifier);
    }).iterator->value->layer().setTexture(WTFMove(texture));
}

void WCRemoteFrameHostLayerManager::removeAllLayersForProcess(WebCore::ProcessIdentifier webProcessIdentifier)
{
    m_layers.removeIf([&](auto& iterator) {
        return iterator.value->webProcessIdentifier() == webProcessIdentifier;
    });
}

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
