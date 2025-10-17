/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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

#include "BufferAndBackendInfo.h"
#include "BufferIdentifierSet.h"
#include "ImageBufferBackendHandle.h"
#include "RemoteImageBufferSetIdentifier.h"
#include "RemoteImageBufferSetProxy.h"
#include <WebCore/FloatRect.h>
#include <WebCore/ImageBuffer.h>
#include <WebCore/PlatformCALayer.h>
#include <WebCore/Region.h>
#include <wtf/MachSendRight.h>
#include <wtf/MonotonicTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/WeakRef.h>

OBJC_CLASS CALayer;
OBJC_CLASS UIView;

// FIXME: Make PlatformCALayerRemote.cpp Objective-C so we can include WebLayer.h here and share the typedef.
namespace WebCore {
class NativeImage;
typedef Vector<WebCore::FloatRect, 5> RepaintRectList;
struct PlatformCALayerDelegatedContents;
struct PlatformCALayerDelegatedContentsFinishedEvent;
}

namespace WebKit {

class PlatformCALayerRemote;
class RemoteLayerBackingStoreCollection;
class RemoteLayerTreeNode;
class RemoteLayerTreeHost;
class ThreadSafeImageBufferSetFlusher;
enum class SwapBuffersDisplayRequirement : uint8_t;
struct PlatformCALayerRemoteDelegatedContents;

enum class BackingStoreNeedsDisplayReason : uint8_t {
    None,
    NoFrontBuffer,
    FrontBufferIsVolatile,
    FrontBufferHasNoSharingHandle,
    HasDirtyRegion,
};

enum class LayerContentsType : uint8_t {
    IOSurface,
    CAMachPort,
    CachedIOSurface,
};

class RemoteLayerBackingStore : public CanMakeWeakPtr<RemoteLayerBackingStore>, public CanMakeCheckedPtr<RemoteLayerBackingStore> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerBackingStore);
    WTF_MAKE_NONCOPYABLE(RemoteLayerBackingStore);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteLayerBackingStore);
public:
    RemoteLayerBackingStore(PlatformCALayerRemote&);
    virtual ~RemoteLayerBackingStore();

    static std::unique_ptr<RemoteLayerBackingStore> createForLayer(PlatformCALayerRemote&);

    enum class Type : bool {
        IOSurface,
        Bitmap
    };

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    enum class IncludeDisplayList : bool { No, Yes };
#endif

    virtual bool isRemoteLayerWithRemoteRenderingBackingStore() const { return false; }
    virtual bool isRemoteLayerWithInProcessRenderingBackingStore() const { return false; }

    enum class ProcessModel : uint8_t { InProcess, Remote };
    virtual ProcessModel processModel() const = 0;
    static ProcessModel processModelForLayer(PlatformCALayerRemote&);

    struct Parameters {
        Type type { Type::Bitmap };
        WebCore::FloatSize size;
        WebCore::DestinationColorSpace colorSpace { WebCore::DestinationColorSpace::SRGB() };
        WebCore::ContentsFormat contentsFormat { WebCore::ContentsFormat::RGBA8 };
        float scale { 1.0f };
        bool isOpaque { false };

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
        IncludeDisplayList includeDisplayList { IncludeDisplayList::No };
#endif

        friend bool operator==(const Parameters&, const Parameters&) = default;
    };

    virtual void ensureBackingStore(const Parameters&);

    void setNeedsDisplay(const WebCore::IntRect);
    void setNeedsDisplay();

    void setDelegatedContents(const PlatformCALayerRemoteDelegatedContents&);

    // Returns true if we need to encode the buffer.
    bool layerWillBeDisplayed();
    bool layerWillBeDisplayedWithRenderingSuppression();
    bool needsDisplay() const;

    bool performDelegatedLayerDisplay();

    void paintContents();
    virtual void prepareToDisplay() = 0;
    virtual void createContextAndPaintContents() = 0;

    virtual std::unique_ptr<ThreadSafeImageBufferSetFlusher> createFlusher(ThreadSafeImageBufferSetFlusher::FlushType = ThreadSafeImageBufferSetFlusher::FlushType::BackendHandlesAndDrawing) = 0;

    WebCore::FloatSize size() const { return m_parameters.size; }
    float scale() const { return m_parameters.scale; }
    WebCore::ContentsFormat contentsFormat() const { return m_parameters.contentsFormat; }
    WebCore::DestinationColorSpace colorSpace() const { return m_parameters.colorSpace; }
    WebCore::ImageBufferPixelFormat pixelFormat() const;
    Type type() const { return m_parameters.type; }
    bool isOpaque() const { return m_parameters.isOpaque; }
    unsigned bytesPerPixel() const;
    bool supportsPartialRepaint() const;
    bool drawingRequiresClearedPixels() const;

    PlatformCALayerRemote& layer() const;

    void encode(IPC::Encoder&) const;

    void enumerateRectsBeingDrawn(WebCore::GraphicsContext&, void (^)(WebCore::FloatRect));

    virtual bool hasFrontBuffer() const = 0;
    virtual bool frontBufferMayBeVolatile() const = 0;

    virtual void encodeBufferAndBackendInfos(IPC::Encoder&) const = 0;

    Vector<std::unique_ptr<ThreadSafeImageBufferSetFlusher>> takePendingFlushers();

    enum class BufferType {
        Front,
        Back,
        SecondaryBack
    };

    const WebCore::Region& dirtyRegion() { return m_dirtyRegion; }
    bool hasEmptyDirtyRegion() const { return m_dirtyRegion.isEmpty() || m_parameters.size.isEmpty(); }

    MonotonicTime lastDisplayTime() const { return m_lastDisplayTime; }

    virtual void clearBackingStore() = 0;

    virtual std::optional<ImageBufferBackendHandle> frontBufferHandle() const = 0;
#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    virtual std::optional<ImageBufferBackendHandle> displayListHandle() const  { return std::nullopt; }
#endif
    virtual std::optional<RemoteImageBufferSetIdentifier> bufferSetIdentifier() const { return std::nullopt; }

    virtual void dump(WTF::TextStream&) const = 0;

    void purgeFrontBufferForTesting();
    void purgeBackBufferForTesting();
    void markFrontBufferVolatileForTesting();

protected:
    RemoteLayerBackingStoreCollection* backingStoreCollection() const;

    void drawInContext(WebCore::GraphicsContext&);

    void dirtyRepaintCounterIfNecessary();

    WebCore::IntRect layerBounds() const;

    WeakRef<PlatformCALayerRemote> m_layer;

    Parameters m_parameters;

    WebCore::Region m_dirtyRegion;

    std::optional<WebCore::IntRect> m_previouslyPaintedRect;

    // FIXME: This should be removed and m_bufferHandle should be used to ref the buffer once ShareableBitmap::Handle
    // can be encoded multiple times. http://webkit.org/b/234169
    std::optional<ImageBufferBackendHandle> m_contentsBufferHandle;
    std::optional<WebCore::RenderingResourceIdentifier> m_contentsRenderingResourceIdentifier;

    Vector<std::unique_ptr<ThreadSafeImageBufferSetFlusher>> m_frontBufferFlushers;

    WebCore::RepaintRectList m_paintingRects;

    MonotonicTime m_lastDisplayTime;
};

// The subset of RemoteLayerBackingStore that gets serialized into the UI
// process, and gets applied to the CALayer.
class RemoteLayerBackingStoreProperties {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerBackingStoreProperties);
    WTF_MAKE_NONCOPYABLE(RemoteLayerBackingStoreProperties);
public:
    RemoteLayerBackingStoreProperties() = default;
    RemoteLayerBackingStoreProperties(RemoteLayerBackingStoreProperties&&) = default;

    void applyBackingStoreToLayer(CALayer *, LayerContentsType, std::optional<WebCore::RenderingResourceIdentifier>, bool replayDynamicContentScalingDisplayListsIntoBackingStore, UIView * hostingView);

    void updateCachedBuffers(RemoteLayerTreeNode&, LayerContentsType);

    const std::optional<ImageBufferBackendHandle>& bufferHandle() const { return m_bufferHandle; };

    bool isOpaque() const { return m_isOpaque; }

    static RetainPtr<id> layerContentsBufferFromBackendHandle(ImageBufferBackendHandle&&, LayerContentsType);

    void dump(WTF::TextStream&) const;

    std::optional<RemoteImageBufferSetIdentifier> bufferSetIdentifier() { return m_bufferSet; }
    void setBackendHandle(BufferSetBackendHandle&);

private:
    friend struct IPC::ArgumentCoder<RemoteLayerBackingStoreProperties, void>;
    std::optional<ImageBufferBackendHandle> m_bufferHandle;
    RetainPtr<id> m_contentsBuffer;

    std::optional<RemoteImageBufferSetIdentifier> m_bufferSet;

    std::optional<BufferAndBackendInfo> m_frontBufferInfo;
    std::optional<BufferAndBackendInfo> m_backBufferInfo;
    std::optional<BufferAndBackendInfo> m_secondaryBackBufferInfo;
    std::optional<WebCore::RenderingResourceIdentifier> m_contentsRenderingResourceIdentifier;

    std::optional<WebCore::IntRect> m_paintedRect;

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    std::optional<ImageBufferBackendHandle> m_displayListBufferHandle;
#endif

    bool m_isOpaque;
    RemoteLayerBackingStore::Type m_type;
};

WTF::TextStream& operator<<(WTF::TextStream&, BackingStoreNeedsDisplayReason);
WTF::TextStream& operator<<(WTF::TextStream&, const RemoteLayerBackingStore&);
WTF::TextStream& operator<<(WTF::TextStream&, const RemoteLayerBackingStoreProperties&);

} // namespace WebKit
