/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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
#include "GraphicsLayerCARemote.h"

#include "ImageBufferBackendHandleSharing.h"
#include "PlatformCAAnimationRemote.h"
#include "PlatformCALayerRemote.h"
#include "PlatformCALayerRemoteHost.h"
#include "RemoteLayerTreeContext.h"
#include "RemoteLayerTreeDrawingAreaProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/GraphicsLayerContentsDisplayDelegate.h>
#include <WebCore/HTMLVideoElement.h>
#include <WebCore/Model.h>
#include <WebCore/PlatformCALayerDelegatedContents.h>
#include <WebCore/PlatformScreen.h>
#include <WebCore/RemoteFrame.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(MODEL_PROCESS)
#include <WebCore/ModelContext.h>
#endif

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(GraphicsLayerCARemote);

GraphicsLayerCARemote::GraphicsLayerCARemote(Type layerType, GraphicsLayerClient& client, RemoteLayerTreeContext& context)
    : GraphicsLayerCA(layerType, client)
    , m_context(&context)
{
    context.graphicsLayerDidEnterContext(*this);
}

GraphicsLayerCARemote::~GraphicsLayerCARemote()
{
    if (RefPtr<RemoteLayerTreeContext> protectedContext = m_context.get())
        protectedContext->graphicsLayerWillLeaveContext(*this);
}

bool GraphicsLayerCARemote::filtersCanBeComposited(const FilterOperations& filters)
{
    return PlatformCALayerRemote::filtersCanBeComposited(filters);
}

Ref<PlatformCALayer> GraphicsLayerCARemote::createPlatformCALayer(PlatformCALayer::LayerType layerType, PlatformCALayerClient* owner)
{
    RELEASE_ASSERT(m_context.get());
    auto result = PlatformCALayerRemote::create(layerType, owner, *m_context);

    if (result->canHaveBackingStore()) {
        auto* localMainFrameView = m_context->webPage().localMainFrameView();
        result->setContentsFormat(screenContentsFormat(localMainFrameView, owner));
    }

    return WTFMove(result);
}

Ref<PlatformCALayer> GraphicsLayerCARemote::createPlatformCALayer(PlatformLayer* platformLayer, PlatformCALayerClient* owner)
{
    RELEASE_ASSERT(m_context.get());
    return PlatformCALayerRemote::create(platformLayer, owner, *m_context);
}

#if ENABLE(MODEL_PROCESS)
Ref<PlatformCALayer> GraphicsLayerCARemote::createPlatformCALayer(Ref<WebCore::ModelContext> modelContext, PlatformCALayerClient* owner)
{
    RELEASE_ASSERT(m_context.get());
    return PlatformCALayerRemote::create(modelContext, owner, *m_context);
}
#endif

#if ENABLE(MODEL_ELEMENT)
Ref<PlatformCALayer> GraphicsLayerCARemote::createPlatformCALayer(Ref<WebCore::Model> model, PlatformCALayerClient* owner)
{
    RELEASE_ASSERT(m_context.get());
    return PlatformCALayerRemote::create(model, owner, *m_context);
}
#endif

Ref<PlatformCALayer> GraphicsLayerCARemote::createPlatformCALayerHost(WebCore::LayerHostingContextIdentifier identifier, PlatformCALayerClient* owner)
{
    RELEASE_ASSERT(m_context.get());
    return PlatformCALayerRemoteHost::create(identifier, owner, *m_context);
}

#if HAVE(AVKIT)
Ref<PlatformCALayer> GraphicsLayerCARemote::createPlatformVideoLayer(WebCore::HTMLVideoElement& videoElement, PlatformCALayerClient* owner)
{
    RELEASE_ASSERT(m_context.get());
    return PlatformCALayerRemote::create(videoElement, owner, *m_context);
}
#endif

Ref<PlatformCAAnimation> GraphicsLayerCARemote::createPlatformCAAnimation(PlatformCAAnimation::AnimationType type, const String& keyPath)
{
    return PlatformCAAnimationRemote::create(type, keyPath);
}

void GraphicsLayerCARemote::moveToContext(RemoteLayerTreeContext& context)
{
    if (RefPtr protectedContext = m_context.get())
        protectedContext->graphicsLayerWillLeaveContext(*this);

    m_context = &context;

    context.graphicsLayerDidEnterContext(*this);
}

Color GraphicsLayerCARemote::pageTiledBackingBorderColor() const
{
    return SRGBA<uint8_t> { 28, 74, 120, 128 }; // remote tile cache layer: navy blue
}

class GraphicsLayerCARemoteAsyncContentsDisplayDelegate : public GraphicsLayerAsyncContentsDisplayDelegate {
public:
    GraphicsLayerCARemoteAsyncContentsDisplayDelegate(IPC::Connection& connection, DrawingAreaIdentifier identifier)
        : m_connection(connection)
        , m_drawingArea(identifier)
    { }

    bool tryCopyToLayer(ImageBuffer& buffer) final
    {
        auto clone = buffer.clone();
        if (!clone)
            return false;

        clone->flushDrawingContext();

        auto* sharing = dynamicDowncast<ImageBufferBackendHandleSharing>(clone->toBackendSharing());
        if (!sharing)
            return false;

        auto backendHandle = sharing->createBackendHandle(SharedMemory::Protection::ReadOnly);
        ASSERT(backendHandle);

        {
            Locker locker { m_surfaceLock };
            m_surfaceBackendHandle = ImageBufferBackendHandle { *backendHandle };
            m_surfaceIdentifier = clone->renderingResourceIdentifier();
        }

        m_connection->send(Messages::RemoteLayerTreeDrawingAreaProxy::AsyncSetLayerContents(*m_layerID, WTFMove(*backendHandle), clone->renderingResourceIdentifier()), m_drawingArea.toUInt64());

        return true;
    }

    void display(PlatformCALayer& layer) final
    {
        Locker locker { m_surfaceLock };
        if (m_surfaceBackendHandle)
            downcast<PlatformCALayerRemote>(layer).setRemoteDelegatedContents({ ImageBufferBackendHandle { *m_surfaceBackendHandle }, { }, std::optional<RenderingResourceIdentifier>(m_surfaceIdentifier) });
    }

    void setDestinationLayerID(WebCore::PlatformLayerIdentifier layerID)
    {
        m_layerID = layerID;
    }

    bool isGraphicsLayerCARemoteAsyncContentsDisplayDelegate() const final { return true; }

private:
    Ref<IPC::Connection> m_connection;
    DrawingAreaIdentifier m_drawingArea;
    Markable<WebCore::PlatformLayerIdentifier> m_layerID;
    Lock m_surfaceLock;
    std::optional<ImageBufferBackendHandle> m_surfaceBackendHandle WTF_GUARDED_BY_LOCK(m_surfaceLock);
    Markable<WebCore::RenderingResourceIdentifier> m_surfaceIdentifier WTF_GUARDED_BY_LOCK(m_surfaceLock);
};

RefPtr<WebCore::GraphicsLayerAsyncContentsDisplayDelegate> GraphicsLayerCARemote::createAsyncContentsDisplayDelegate(GraphicsLayerAsyncContentsDisplayDelegate* existing)
{
    RefPtr protectedContext = m_context.get();
    if (!protectedContext || !protectedContext->drawingAreaIdentifier() || !WebProcess::singleton().parentProcessConnection())
        return nullptr;

    RefPtr<GraphicsLayerCARemoteAsyncContentsDisplayDelegate> delegate;
    if (existing && existing->isGraphicsLayerCARemoteAsyncContentsDisplayDelegate())
        delegate = static_cast<GraphicsLayerCARemoteAsyncContentsDisplayDelegate*>(existing);

    if (!delegate) {
        ASSERT(!existing);
        delegate = adoptRef(new GraphicsLayerCARemoteAsyncContentsDisplayDelegate(*WebProcess::singleton().parentProcessConnection(), *protectedContext->drawingAreaIdentifier()));
    }

    auto layerID = setContentsToAsyncDisplayDelegate(delegate, ContentsLayerPurpose::Canvas);

    delegate->setDestinationLayerID(layerID);
    return delegate;
}

bool GraphicsLayerCARemote::shouldDirectlyCompositeImageBuffer(ImageBuffer* image) const
{
    return !!dynamicDowncast<ImageBufferBackendHandleSharing>(image->toBackendSharing());
}

void GraphicsLayerCARemote::setLayerContentsToImageBuffer(PlatformCALayer* layer, ImageBuffer* image)
{
    if (!image)
        return;

    image->flushDrawingContextAsync();

    auto* sharing = dynamicDowncast<ImageBufferBackendHandleSharing>(image->toBackendSharing());
    if (!sharing)
        return;

    auto backendHandle = sharing->createBackendHandle(SharedMemory::Protection::ReadOnly);
    ASSERT(backendHandle);

    layer->setAcceleratesDrawing(true);
    downcast<PlatformCALayerRemote>(layer)->setRemoteDelegatedContents({ ImageBufferBackendHandle { *backendHandle }, { }, std::nullopt  });
}

GraphicsLayer::LayerMode GraphicsLayerCARemote::layerMode() const
{
    if (m_context && m_context->layerHostingMode() == LayerHostingMode::InProcess)
        return GraphicsLayer::LayerMode::PlatformLayer;
    return GraphicsLayer::LayerMode::LayerHostingContextId;
}

} // namespace WebKit
