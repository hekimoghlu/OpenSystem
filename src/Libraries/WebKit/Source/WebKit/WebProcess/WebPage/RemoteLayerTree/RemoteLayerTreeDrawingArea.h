/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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

#include "CallbackID.h"
#include "DrawingArea.h"
#include "GraphicsLayerCARemote.h"
#include "RemoteLayerTreeTransaction.h"
#include <WebCore/AnimationFrameRate.h>
#include <WebCore/GraphicsLayerClient.h>
#include <WebCore/Timer.h>
#include <atomic>
#include <dispatch/dispatch.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class PlatformCALayer;
class ThreadSafeImageBufferFlusher;
class TiledBacking;
}

namespace WebKit {

class RemoteLayerTreeContext;

class RemoteLayerTreeDrawingArea : public DrawingArea, public WebCore::GraphicsLayerClient {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerTreeDrawingArea);
public:
    static RefPtr<RemoteLayerTreeDrawingArea> create(WebPage& webPage, const WebPageCreationParameters& parameters)
    {
        return adoptRef(*new RemoteLayerTreeDrawingArea(webPage, parameters));
    }

    virtual ~RemoteLayerTreeDrawingArea();

    TransactionID nextTransactionID() const { return m_currentTransactionID.next(); }
    TransactionID lastCommittedTransactionID() const { return m_currentTransactionID; }

    bool displayDidRefreshIsPending() const { return m_waitingForBackingStoreSwap; }

    void gpuProcessConnectionWasDestroyed();

protected:
    RemoteLayerTreeDrawingArea(WebPage&, const WebPageCreationParameters&);

    void updateRendering();

private:
    // DrawingArea
    void setNeedsDisplay() final;
    void setNeedsDisplayInRect(const WebCore::IntRect&) final;
    void scroll(const WebCore::IntRect& scrollRect, const WebCore::IntSize& scrollDelta) final;
    void updateGeometry(const WebCore::IntSize& viewSize, bool flushSynchronously, const WTF::MachSendRight& fencePort, CompletionHandler<void()>&&) final;

    WebCore::GraphicsLayerFactory* graphicsLayerFactory() final;
    void setRootCompositingLayer(WebCore::Frame&, WebCore::GraphicsLayer*) final;
    void addRootFrame(WebCore::FrameIdentifier) final;
    void removeRootFrame(WebCore::FrameIdentifier) final;
    void triggerRenderingUpdate() final;
    bool scheduleRenderingUpdate() final;
    void renderingUpdateFramesPerSecondChanged() final;
    void attachViewOverlayGraphicsLayer(WebCore::FrameIdentifier, WebCore::GraphicsLayer*) final;

    void dispatchAfterEnsuringDrawing(IPC::AsyncReplyID) final;
    virtual void willCommitLayerTree(RemoteLayerTreeTransaction&) { }

    RefPtr<WebCore::DisplayRefreshMonitor> createDisplayRefreshMonitor(WebCore::PlatformDisplayID) final;
    void setPreferredFramesPerSecond(WebCore::FramesPerSecond);

    bool shouldUseTiledBackingForFrameView(const WebCore::LocalFrameView&) const final;

    void updatePreferences(const WebPreferencesStore&) final;

    bool supportsAsyncScrolling() const final { return true; }
    bool usesDelegatedPageScaling() const override { return true; }
    WebCore::DelegatedScrollingMode delegatedScrollingMode() const override;

    void setLayerTreeStateIsFrozen(bool) final;
    bool layerTreeStateIsFrozen() const final { return m_isRenderingSuspended; }

    void updateRenderingWithForcedRepaint() final;
    void updateRenderingWithForcedRepaintAsync(WebPage&, CompletionHandler<void()>&&) final;

    void setViewExposedRect(std::optional<WebCore::FloatRect>) final;
    std::optional<WebCore::FloatRect> viewExposedRect() const final { return m_viewExposedRect; }

    void acceleratedAnimationDidStart(WebCore::PlatformLayerIdentifier, const String& key, MonotonicTime startTime) final;
    void acceleratedAnimationDidEnd(WebCore::PlatformLayerIdentifier, const String& key) final;

    WebCore::FloatRect exposedContentRect() const final;
    void setExposedContentRect(const WebCore::FloatRect&) final;

    void displayDidRefresh() final;

    void setDeviceScaleFactor(float, CompletionHandler<void()>&&) final;

    void mainFrameContentSizeChanged(WebCore::FrameIdentifier, const WebCore::IntSize&) override;

    void activityStateDidChange(OptionSet<WebCore::ActivityState> changed, ActivityStateChangeID, CompletionHandler<void()>&&) final;

    bool addMilestonesToDispatch(OptionSet<WebCore::LayoutMilestone>) final;

    void updateRootLayers();

    void addCommitHandlers();
    void startRenderingUpdateTimer();
    void didCompleteRenderingUpdateDisplayFlush(bool flushSucceeded);

    TransactionID takeNextTransactionID() { return m_currentTransactionID.increment(); }

    void tryMarkLayersVolatile(CompletionHandler<void(bool succeeded)>&&) final;

    void adoptLayersFromDrawingArea(DrawingArea&) final;

    void setNextRenderingUpdateRequiresSynchronousImageDecoding() final;

    void scheduleRenderingUpdateTimerFired();

    class BackingStoreFlusher : public ThreadSafeRefCounted<BackingStoreFlusher> {
    public:
        static Ref<BackingStoreFlusher> create(Ref<IPC::Connection>&&);

        // Returns true when flush succeeds. False if it failed, for example due to timeout.
        bool flush(UniqueRef<IPC::Encoder>&&, Vector<std::unique_ptr<ThreadSafeImageBufferSetFlusher>>&&);

        bool hasPendingFlush() const { return m_hasPendingFlush; }
        void markHasPendingFlush()
        {
            bool hadPendingFlush = m_hasPendingFlush.exchange(true);
            RELEASE_ASSERT(!hadPendingFlush);
        }

    private:
        explicit BackingStoreFlusher(Ref<IPC::Connection>&&);

        Ref<IPC::Connection> m_connection;
        std::atomic<bool> m_hasPendingFlush { false };
    };

    Ref<RemoteLayerTreeContext> m_remoteLayerTreeContext;
    
    struct RootLayerInfo {
        Ref<WebCore::GraphicsLayer> layer;
        RefPtr<WebCore::GraphicsLayer> contentLayer;
        RefPtr<WebCore::GraphicsLayer> viewOverlayRootLayer;
        WebCore::FrameIdentifier frameID;
    };
    RootLayerInfo* rootLayerInfoWithFrameIdentifier(WebCore::FrameIdentifier);

    // GraphicsLayerClient
#if HAVE(HDR_SUPPORT)
    bool hdrForImagesEnabled() const override;
#endif

    Vector<RootLayerInfo, 1> m_rootLayers;

    std::optional<WebCore::FloatRect> m_viewExposedRect;

    WebCore::Timer m_updateRenderingTimer;
    bool m_isRenderingSuspended { false };
    bool m_hasDeferredRenderingUpdate { false };
    bool m_inUpdateRendering { false };

    bool m_waitingForBackingStoreSwap { false };
    bool m_deferredRenderingUpdateWhileWaitingForBackingStoreSwap { false };

    Ref<WorkQueue> m_commitQueue;
    RefPtr<BackingStoreFlusher> m_backingStoreFlusher;

    TransactionID m_currentTransactionID;
    Vector<IPC::AsyncReplyID> m_pendingCallbackIDs;
    ActivityStateChangeID m_activityStateChangeID { ActivityStateChangeAsynchronous };

    OptionSet<WebCore::LayoutMilestone> m_pendingNewlyReachedPaintingMilestones;

    WebCore::Timer m_scheduleRenderingTimer;
    std::optional<WebCore::FramesPerSecond> m_preferredFramesPerSecond;
    Seconds m_preferredRenderingUpdateInterval;
    bool m_isScheduled { false };
};

inline bool RemoteLayerTreeDrawingArea::addMilestonesToDispatch(OptionSet<WebCore::LayoutMilestone> paintMilestones)
{
    m_pendingNewlyReachedPaintingMilestones.add(paintMilestones);
    return true;
}

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_DRAWING_AREA(RemoteLayerTreeDrawingArea, DrawingAreaType::RemoteLayerTree)
