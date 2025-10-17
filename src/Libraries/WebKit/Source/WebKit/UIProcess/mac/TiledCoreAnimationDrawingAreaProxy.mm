/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
#import "config.h"
#import "TiledCoreAnimationDrawingAreaProxy.h"

#if !PLATFORM(IOS_FAMILY)

#import "DrawingAreaMessages.h"
#import "DrawingAreaProxyMessages.h"
#import "LayerTreeContext.h"
#import "MessageSenderInlines.h"
#import "WebPageProxy.h"
#import "WebPageProxyMessages.h"
#import "WebProcessProxy.h"
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/BlockPtr.h>
#import <wtf/MachSendRight.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace IPC;
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(TiledCoreAnimationDrawingAreaProxy);

Ref<TiledCoreAnimationDrawingAreaProxy> TiledCoreAnimationDrawingAreaProxy::create(WebPageProxy& page, WebProcessProxy& webProcessProxy)
{
    return adoptRef(*new TiledCoreAnimationDrawingAreaProxy(page, webProcessProxy));
}

TiledCoreAnimationDrawingAreaProxy::TiledCoreAnimationDrawingAreaProxy(WebPageProxy& webPageProxy, WebProcessProxy& webProcessProxy)
    : DrawingAreaProxy(DrawingAreaType::TiledCoreAnimation, webPageProxy, webProcessProxy)
{
}

TiledCoreAnimationDrawingAreaProxy::~TiledCoreAnimationDrawingAreaProxy() = default;

void TiledCoreAnimationDrawingAreaProxy::deviceScaleFactorDidChange(CompletionHandler<void()>&& completionHandler)
{
    if (RefPtr webPageProxy = m_webPageProxy.get())
        sendWithAsyncReply(Messages::DrawingArea::SetDeviceScaleFactor(webPageProxy->deviceScaleFactor()), WTFMove(completionHandler));
}

void TiledCoreAnimationDrawingAreaProxy::sizeDidChange()
{
    RefPtr page = m_webPageProxy.get();
    if (!page || !page->hasRunningProcess())
        return;

    // We only want one UpdateGeometry message in flight at once, so if we've already sent one but
    // haven't yet received the reply we'll just return early here.
    if (m_isWaitingForDidUpdateGeometry)
        return;

    sendUpdateGeometry();
}

void TiledCoreAnimationDrawingAreaProxy::colorSpaceDidChange()
{
    if (RefPtr page = m_webPageProxy.get())
        send(Messages::DrawingArea::SetColorSpace(page->colorSpace()));
}

void TiledCoreAnimationDrawingAreaProxy::minimumSizeForAutoLayoutDidChange()
{
    RefPtr page = m_webPageProxy.get();
    if (!page || !page->hasRunningProcess())
        return;

    // We only want one UpdateGeometry message in flight at once, so if we've already sent one but
    // haven't yet received the reply we'll just return early here.
    if (m_isWaitingForDidUpdateGeometry)
        return;

    sendUpdateGeometry();
}

void TiledCoreAnimationDrawingAreaProxy::sizeToContentAutoSizeMaximumSizeDidChange()
{
    RefPtr page = m_webPageProxy.get();
    if (!page || !page->hasRunningProcess())
        return;

    // We only want one UpdateGeometry message in flight at once, so if we've already sent one but
    // haven't yet received the reply we'll just return early here.
    if (m_isWaitingForDidUpdateGeometry)
        return;

    sendUpdateGeometry();
}

void TiledCoreAnimationDrawingAreaProxy::enterAcceleratedCompositingMode(uint64_t backingStoreStateID, const LayerTreeContext& layerTreeContext)
{
    if (RefPtr page = m_webPageProxy.get())
        page->enterAcceleratedCompositingMode(layerTreeContext);
}

void TiledCoreAnimationDrawingAreaProxy::updateAcceleratedCompositingMode(uint64_t backingStoreStateID, const LayerTreeContext& layerTreeContext)
{
    if (RefPtr page = m_webPageProxy.get())
        page->updateAcceleratedCompositingMode(layerTreeContext);
}

void TiledCoreAnimationDrawingAreaProxy::didFirstLayerFlush(uint64_t /* backingStoreStateID */, const LayerTreeContext& layerTreeContext)
{
    if (RefPtr page = m_webPageProxy.get())
        page->didFirstLayerFlush(layerTreeContext);
}

void TiledCoreAnimationDrawingAreaProxy::didUpdateGeometry()
{
    ASSERT(m_isWaitingForDidUpdateGeometry);

    m_isWaitingForDidUpdateGeometry = false;

    RefPtr page = m_webPageProxy.get();
    if (!page)
        return;

    IntSize minimumSizeForAutoLayout = page->minimumSizeForAutoLayout();
    IntSize sizeToContentAutoSizeMaximumSize = page->sizeToContentAutoSizeMaximumSize();

    // If the WKView was resized while we were waiting for a DidUpdateGeometry reply from the web process,
    // we need to resend the new size here.
    if (m_lastSentSize != m_size || m_lastSentMinimumSizeForAutoLayout != minimumSizeForAutoLayout || m_lastSentSizeToContentAutoSizeMaximumSize != sizeToContentAutoSizeMaximumSize)
        sendUpdateGeometry();
}

void TiledCoreAnimationDrawingAreaProxy::waitForDidUpdateActivityState(ActivityStateChangeID)
{
    RefPtr page = m_webPageProxy.get();
    if (!page)
        return;

    Seconds activityStateUpdateTimeout = Seconds::fromMilliseconds(250);
    protectedWebProcessProxy()->protectedConnection()->waitForAndDispatchImmediately<Messages::WebPageProxy::DidUpdateActivityState>(page->webPageIDInMainFrameProcess(), activityStateUpdateTimeout, IPC::WaitForOption::InterruptWaitingIfSyncMessageArrives);
}

void TiledCoreAnimationDrawingAreaProxy::willSendUpdateGeometry()
{
    RefPtr page = m_webPageProxy.get();
    if (!page)
        return;

    m_lastSentMinimumSizeForAutoLayout = page->minimumSizeForAutoLayout();
    m_lastSentSizeToContentAutoSizeMaximumSize = page->sizeToContentAutoSizeMaximumSize();
    m_lastSentSize = m_size;
    m_isWaitingForDidUpdateGeometry = true;
}

MachSendRight TiledCoreAnimationDrawingAreaProxy::createFence()
{
    RefPtr page = m_webPageProxy.get();
    if (!page)
        return MachSendRight();

    RetainPtr<CAContext> rootLayerContext = [page->acceleratedCompositingRootLayer() context];
    if (!rootLayerContext)
        return MachSendRight();

    // Don't fence if we don't have a connection, because the message
    // will likely get dropped on the floor (if the Web process is terminated)
    // or queued up until process launch completes, and there's nothing useful
    // to synchronize in these cases.
    if (!m_webProcessProxy->hasConnection())
        return MachSendRight();

    Ref connection = m_webProcessProxy->connection();

    // Don't fence if we have incoming synchronous messages, because we may not
    // be able to reply to the message until the fence times out.
    if (connection->hasIncomingSyncMessage())
        return MachSendRight();

    MachSendRight fencePort = MachSendRight::adopt([rootLayerContext createFencePort]);

    // Invalidate the fence if a synchronous message arrives while it's installed,
    // because we won't be able to reply during the fence-wait.
    uint64_t callbackID = connection->installIncomingSyncMessageCallback([rootLayerContext] {
        [rootLayerContext invalidateFences];
    });
    [CATransaction addCommitHandler:[callbackID, connection = WTFMove(connection)] () mutable {
        connection->uninstallIncomingSyncMessageCallback(callbackID);
    } forPhase:kCATransactionPhasePostCommit];

    return fencePort;
}

void TiledCoreAnimationDrawingAreaProxy::sendUpdateGeometry()
{
    ASSERT(!m_isWaitingForDidUpdateGeometry);

    willSendUpdateGeometry();
    sendWithAsyncReply(Messages::DrawingArea::UpdateGeometry(m_size, true /* flushSynchronously */, createFence()), [weakThis = WeakPtr { *this }] {
        if (!weakThis)
            return;
        weakThis->didUpdateGeometry();
    });
}

void TiledCoreAnimationDrawingAreaProxy::adjustTransientZoom(double scale, FloatPoint origin)
{
    send(Messages::DrawingArea::AdjustTransientZoom(scale, origin));
}

void TiledCoreAnimationDrawingAreaProxy::commitTransientZoom(double scale, FloatPoint origin)
{
    sendWithAsyncReply(Messages::DrawingArea::CommitTransientZoom(scale, origin), [] { });
}

void TiledCoreAnimationDrawingAreaProxy::dispatchPresentationCallbacksAfterFlushingLayers(IPC::Connection& connection, Vector<IPC::AsyncReplyID>&& callbackIDs)
{
    for (auto& callbackID : callbackIDs) {
        if (auto callback = connection.takeAsyncReplyHandler(callbackID))
            callback(nullptr);
    }
}

std::optional<WebCore::FramesPerSecond> TiledCoreAnimationDrawingAreaProxy::displayNominalFramesPerSecond()
{
    if (!m_webPageProxy || !m_webPageProxy->displayID())
        return std::nullopt;
    return protectedWebProcessProxy()->nominalFramesPerSecondForDisplay(*m_webPageProxy->displayID());
}

} // namespace WebKit

#endif // !PLATFORM(IOS_FAMILY)
