/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#include "WCScene.h"

#if USE(GRAPHICS_LAYER_WC)

#include "RemoteGraphicsContextGL.h"
#include "WCContentBuffer.h"
#include "WCContentBufferManager.h"
#include "WCRemoteFrameHostLayerManager.h"
#include "WCSceneContext.h"
#include "WCUpdateInfo.h"
#include <WebCore/BitmapTexture.h>
#include <WebCore/ShareableBitmap.h>
#include <WebCore/TextureMapper.h>
#include <WebCore/TextureMapperGLHeaders.h>
#include <WebCore/TextureMapperLayer.h>
#include <WebCore/TextureMapperPlatformLayer.h>
#include <WebCore/TextureMapperSparseBackingStore.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WCScene);

struct WCScene::Layer final : public WCContentBuffer::Client {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WCScene::Layer);
public:
    Layer() = default;
    ~Layer()
    {
        if (contentBuffer)
            contentBuffer->setClient(nullptr);
        if (hostIdentifier)
            WCRemoteFrameHostLayerManager::singleton().releaseRemoteFrameHostLayer(*hostIdentifier);
    }

    // WCContentBuffer::Client
    void platformLayerWillBeDestroyed() override
    {
        contentBuffer = nullptr;
        texmapLayer.setContentsLayer(nullptr);
    }

    WebCore::TextureMapperLayer texmapLayer;
    std::unique_ptr<WebCore::TextureMapperSparseBackingStore> backingStore;
    std::unique_ptr<WebCore::TextureMapperLayer> backdropLayer;
    WCContentBuffer* contentBuffer { nullptr };
    Markable<WebCore::LayerHostingContextIdentifier> hostIdentifier;
};

void WCScene::initialize(WCSceneContext& context)
{
    // The creation of the TextureMapper needs an active OpenGL context.
    m_context = &context;
    if (!m_context->makeContextCurrent())
        return;
    m_textureMapper = m_context->createTextureMapper();
}

WCScene::WCScene(WebCore::ProcessIdentifier webProcessIdentifier, bool usesOffscreenRendering)
    : m_webProcessIdentifier(webProcessIdentifier)
    , m_usesOffscreenRendering(usesOffscreenRendering)
{
}

WCScene::~WCScene()
{
    if (!m_context->makeContextCurrent())
        return;
    m_textureMapper = nullptr;
}

std::optional<UpdateInfo> WCScene::update(WCUpdateInfo&& update)
{
    if (!m_context->makeContextCurrent())
        return std::nullopt;
    m_textureMapper->releaseUnusedTexturesNow();

    for (auto id : update.addedLayers) {
        auto layer = makeUnique<Layer>();
        m_layers.add(id, WTFMove(layer));
    }

    for (auto& layerUpdate : update.changedLayers) {
        auto layer = m_layers.get(layerUpdate.id);
        if (layerUpdate.changes & WCLayerChange::Children) {
            layer->texmapLayer.setChildren(WTF::map(layerUpdate.children, [&](auto id) {
                return &m_layers.get(id)->texmapLayer;
            }));
        }
        if (layerUpdate.changes & WCLayerChange::MaskLayer) {
            WebCore::TextureMapperLayer* maskLayer = nullptr;
            if (layerUpdate.maskLayer)
                maskLayer = &m_layers.get(*layerUpdate.maskLayer)->texmapLayer;
            layer->texmapLayer.setMaskLayer(maskLayer);
        }
        if (layerUpdate.changes & WCLayerChange::ReplicaLayer) {
            WebCore::TextureMapperLayer* replicaLayer = nullptr;
            if (layerUpdate.replicaLayer)
                replicaLayer = &m_layers.get(*layerUpdate.replicaLayer)->texmapLayer;
            layer->texmapLayer.setReplicaLayer(replicaLayer);
        }
        if (layerUpdate.changes & WCLayerChange::Position)
            layer->texmapLayer.setPosition(layerUpdate.position);
        if (layerUpdate.changes & WCLayerChange::AnchorPoint)
            layer->texmapLayer.setAnchorPoint(layerUpdate.anchorPoint);
        if (layerUpdate.changes & WCLayerChange::Size)
            layer->texmapLayer.setSize(layerUpdate.size);
        if (layerUpdate.changes & WCLayerChange::BoundsOrigin)
            layer->texmapLayer.setBoundsOrigin(layerUpdate.boundsOrigin);
        if (layerUpdate.changes & WCLayerChange::Preserves3D)
            layer->texmapLayer.setPreserves3D(layerUpdate.preserves3D);
        if (layerUpdate.changes & WCLayerChange::ContentsRect)
            layer->texmapLayer.setContentsRect(layerUpdate.contentsRect);
        if (layerUpdate.changes & WCLayerChange::ContentsClippingRect)
            layer->texmapLayer.setContentsClippingRect(layerUpdate.contentsClippingRect);
        if (layerUpdate.changes & WCLayerChange::ContentsRectClipsDescendants)
            layer->texmapLayer.setContentsRectClipsDescendants(layerUpdate.contentsRectClipsDescendants);
        if (layerUpdate.changes & WCLayerChange::ContentsVisible)
            layer->texmapLayer.setContentsVisible(layerUpdate.contentsVisible);
        if (layerUpdate.changes & WCLayerChange::BackfaceVisibility)
            layer->texmapLayer.setBackfaceVisibility(layerUpdate.backfaceVisibility);
        if (layerUpdate.changes & WCLayerChange::MasksToBounds)
            layer->texmapLayer.setMasksToBounds(layerUpdate.masksToBounds);
        if (layerUpdate.changes & WCLayerChange::Background) {
            if (layerUpdate.background.hasBackingStore) {
                if (!layer->backingStore) {
                    layer->backingStore = makeUnique<WebCore::TextureMapperSparseBackingStore>();
                    auto& backingStore = *layer->backingStore;
                    layer->texmapLayer.setBackgroundColor({ });
                    layer->texmapLayer.setBackingStore(&backingStore);
                }
                auto& backingStore = *layer->backingStore;
                backingStore.setSize(layerUpdate.background.backingStoreSize);
                for (auto& tileUpdate : layerUpdate.background.tileUpdates) {
                    if (tileUpdate.willRemove)
                        backingStore.removeTile(tileUpdate.index);
                    else {
                        auto bitmap = tileUpdate.backingStore.bitmap();
                        if (bitmap) {
                            auto image = bitmap->createImage();
                            backingStore.updateContents(tileUpdate.index, *image, tileUpdate.dirtyRect);
                        }
                    }
                }
            } else {
                layer->texmapLayer.setBackgroundColor(layerUpdate.background.color);
                layer->texmapLayer.setBackingStore(nullptr);
                layer->backingStore = nullptr;
            }
        }
        if (layerUpdate.changes & WCLayerChange::SolidColor)
            layer->texmapLayer.setSolidColor(layerUpdate.solidColor);
        if (layerUpdate.changes & WCLayerChange::ShowDebugBorder)
            layer->texmapLayer.setShowDebugBorder(layerUpdate.showDebugBorder);
        if (layerUpdate.changes & WCLayerChange::DebugBorderColor)
            layer->texmapLayer.setDebugBorderColor(layerUpdate.debugBorderColor);
        if (layerUpdate.changes & WCLayerChange::DebugBorderWidth)
            layer->texmapLayer.setDebugBorderWidth(layerUpdate.debugBorderWidth);
        if (layerUpdate.changes & WCLayerChange::ShowRepaintCounter)
            layer->texmapLayer.setShowRepaintCounter(layerUpdate.showRepaintCounter);
        if (layerUpdate.changes & WCLayerChange::RepaintCount)
            layer->texmapLayer.setRepaintCount(layerUpdate.repaintCount);
        if (layerUpdate.changes & WCLayerChange::Opacity)
            layer->texmapLayer.setOpacity(layerUpdate.opacity);
        if (layerUpdate.changes & WCLayerChange::Transform)
            layer->texmapLayer.setTransform(layerUpdate.transform);
        if (layerUpdate.changes & WCLayerChange::ChildrenTransform)
            layer->texmapLayer.setChildrenTransform(layerUpdate.childrenTransform);
        if (layerUpdate.changes & WCLayerChange::Filters)
            layer->texmapLayer.setFilters(layerUpdate.filters);
        if (layerUpdate.changes & WCLayerChange::BackdropFilters || layerUpdate.changes & WCLayerChange::BackdropFiltersRect) {
            if (!layer->backdropLayer) {
                layer->backdropLayer = makeUnique<WebCore::TextureMapperLayer>();
                layer->backdropLayer->setAnchorPoint({ });
                layer->backdropLayer->setContentsVisible(true);
                layer->backdropLayer->setMasksToBounds(true);
            }
        }
        if (layerUpdate.changes & WCLayerChange::BackdropFilters) {
            if (layerUpdate.backdropFilters.isEmpty())
                layer->texmapLayer.setBackdropLayer(nullptr);
            else {
                layer->backdropLayer->setFilters(layerUpdate.backdropFilters);
                layer->texmapLayer.setBackdropLayer(layer->backdropLayer.get());
            }
        }
        if (layerUpdate.changes & WCLayerChange::BackdropFiltersRect) {
            layer->backdropLayer->setSize(layerUpdate.backdropFiltersRect.rect().size());
            layer->backdropLayer->setPosition(layerUpdate.backdropFiltersRect.rect().location());
            layer->texmapLayer.setBackdropFiltersRect(layerUpdate.backdropFiltersRect);
        }
        if (layerUpdate.changes & WCLayerChange::PlatformLayer) {
            if (!layerUpdate.platformLayer.hasLayer) {
                if (layer->contentBuffer) {
                    layer->contentBuffer->setClient(nullptr);
                    layer->contentBuffer = nullptr;
                }
                layer->texmapLayer.setContentsLayer(nullptr);
            } else {
                WCContentBuffer* contentBuffer = nullptr;
                for (auto identifier : layerUpdate.platformLayer.identifiers)
                    contentBuffer = WCContentBufferManager::singleton().releaseContentBufferIdentifier(m_webProcessIdentifier, identifier);
                if (contentBuffer) {
                    if (layer->contentBuffer)
                        layer->contentBuffer->setClient(nullptr);
                    contentBuffer->setClient(layer);
                    layer->contentBuffer = contentBuffer;
                    layer->texmapLayer.setContentsLayer(contentBuffer->platformLayer());
                }
            }
        }
        if (layerUpdate.changes & WCLayerChange::RemoteFrame) {
            if (layerUpdate.hostIdentifier) {
                auto platformLayer = WCRemoteFrameHostLayerManager::singleton().acquireRemoteFrameHostLayer(*layerUpdate.hostIdentifier, m_webProcessIdentifier);
                layer->texmapLayer.setContentsLayer(platformLayer);
            } else {
                ASSERT(layer->hostIdentifier);
                WCRemoteFrameHostLayerManager::singleton().releaseRemoteFrameHostLayer(*layer->hostIdentifier);
            }
            layer->hostIdentifier = layerUpdate.hostIdentifier;
        }
    }

    for (auto id : update.removedLayers)
        m_layers.remove(id);

    auto rootLayer = &m_layers.get(*update.rootLayer)->texmapLayer;
    rootLayer->applyAnimationsRecursively(MonotonicTime::now());

    WebCore::BitmapTexture* surface = nullptr;
    RefPtr<WebCore::BitmapTexture> texture;
    bool showFPS = true;
    bool readPixel = false;
    RefPtr<WebCore::ShareableBitmap> bitmap;

    if (update.remoteContextHostedIdentifier) {
        showFPS = false;
        texture = m_textureMapper->acquireTextureFromPool(update.viewport, { WebCore::BitmapTexture::Flags::SupportsAlpha });
        surface = texture.get();
    } else if (m_usesOffscreenRendering) {
        readPixel = true;
        texture = m_textureMapper->acquireTextureFromPool(update.viewport, { WebCore::BitmapTexture::Flags::SupportsAlpha });
        surface = texture.get();
    } else
        glViewport(0, 0, update.viewport.width(), update.viewport.height());

    m_textureMapper->beginPainting(WebCore::TextureMapper::FlipY::No, surface);
    rootLayer->paint(*m_textureMapper);
    if (showFPS)
        m_fpsCounter.updateFPSAndDisplay(*m_textureMapper);
    if (readPixel) {
        bitmap = WebCore::ShareableBitmap::create({ update.viewport });
        glReadPixels(0, 0, update.viewport.width(), update.viewport.height(), GL_BGRA, GL_UNSIGNED_BYTE, bitmap->mutableSpan().data());
    }
    m_textureMapper->endPainting();

    std::optional<UpdateInfo> result;
    if (update.remoteContextHostedIdentifier)
        WCRemoteFrameHostLayerManager::singleton().updateTexture(*update.remoteContextHostedIdentifier, m_webProcessIdentifier, WTFMove(texture));
    else if (m_usesOffscreenRendering) {
        if (auto handle = bitmap->createHandle()) {
            result.emplace();
            result->viewSize = update.viewport;
            result->deviceScaleFactor = 1;
            result->updateScaleFactor = 1;
            WebCore::IntRect viewport = { { }, update.viewport };
            result->updateRectBounds = viewport;
            result->updateRects.append(viewport);
            result->bitmapHandle = WTFMove(*handle);
        }
    } else
        m_context->swapBuffers();
    return result;
}

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
