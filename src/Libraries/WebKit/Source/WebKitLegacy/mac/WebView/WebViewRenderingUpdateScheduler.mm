/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
#import "WebViewRenderingUpdateScheduler.h"

#import "WebViewInternal.h"
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/TZoneMallocInlines.h>

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WebCoreThread.h>
#import <WebCore/WebCoreThreadInternal.h>
#import <wtf/RuntimeApplicationChecks.h>
#endif

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebViewRenderingUpdateScheduler);

WebViewRenderingUpdateScheduler::WebViewRenderingUpdateScheduler(WebView* webView)
    : m_webView(webView)
{
    ASSERT(isMainThread());
    ASSERT_ARG(webView, webView);

    m_renderingUpdateRunLoopObserver = makeUnique<WebCore::RunLoopObserver>(WebCore::RunLoopObserver::WellKnownOrder::RenderingUpdate, [weakThis = WeakPtr { this }] {
#if PLATFORM(IOS_FAMILY)
        // Normally the layer flush callback happens before the web lock auto-unlock observer runs.
        // However if the flush is rescheduled from the callback it may get pushed past it, to the next cycle.
        WebThreadLock();
#endif
        CheckedPtr checkedThis = weakThis.get();
        if (!checkedThis)
            return;
        checkedThis->renderingUpdateRunLoopObserverCallback();
    });

    m_postRenderingUpdateRunLoopObserver = makeUnique<WebCore::RunLoopObserver>(WebCore::RunLoopObserver::WellKnownOrder::PostRenderingUpdate, [weakThis = WeakPtr { this }] {
#if PLATFORM(IOS_FAMILY)
        WebThreadLock();
#endif
        CheckedPtr checkedThis = weakThis.get();
        if (!checkedThis)
            return;
        checkedThis->postRenderingUpdateCallback();
    });
}

WebViewRenderingUpdateScheduler::~WebViewRenderingUpdateScheduler() = default;

void WebViewRenderingUpdateScheduler::scheduleRenderingUpdate()
{
    if (m_insideCallback)
        m_rescheduledInsideCallback = true;

    m_renderingUpdateRunLoopObserver->schedule();
}

void WebViewRenderingUpdateScheduler::invalidate()
{
    ASSERT(isMainThread());
    m_webView = nullptr;
    m_renderingUpdateRunLoopObserver->invalidate();
    m_postRenderingUpdateRunLoopObserver->invalidate();
}

void WebViewRenderingUpdateScheduler::didCompleteRenderingUpdateDisplay()
{
    m_haveRegisteredCommitHandlers = false;
    schedulePostRenderingUpdate();
}

void WebViewRenderingUpdateScheduler::schedulePostRenderingUpdate()
{
    m_postRenderingUpdateRunLoopObserver->schedule();
}

void WebViewRenderingUpdateScheduler::registerCACommitHandlers()
{
    if (m_haveRegisteredCommitHandlers)
        return;

    WebView* webView = m_webView;
    [CATransaction addCommitHandler:^{
        [webView _willStartRenderingUpdateDisplay];
    } forPhase:kCATransactionPhasePreLayout];

    [CATransaction addCommitHandler:^{
        [webView _didCompleteRenderingUpdateDisplay];
    } forPhase:kCATransactionPhasePostCommit];
    
    m_haveRegisteredCommitHandlers = true;
}

void WebViewRenderingUpdateScheduler::renderingUpdateRunLoopObserverCallback()
{
    SetForScope insideCallbackScope(m_insideCallback, true);
    m_rescheduledInsideCallback = false;

    updateRendering();
    registerCACommitHandlers();

    if (!m_rescheduledInsideCallback)
        m_renderingUpdateRunLoopObserver->invalidate();
}

void WebViewRenderingUpdateScheduler::postRenderingUpdateCallback()
{
    @autoreleasepool {
        [m_webView _didCompleteRenderingFrame];
        m_postRenderingUpdateRunLoopObserver->invalidate();
    }
}

/*
    Note: Much of the following is obsolete.
    
    The order of events with compositing updates is this:

   Start of runloop                                        End of runloop
        |                                                       |
      --|-------------------------------------------------------|--
           ^         ^                                        ^
           |         |                                        |
    NSWindow update, |                                     CA commit
     NSView drawing  |
        flush        |
                layerSyncRunLoopObserverCallBack

    To avoid flashing, we have to ensure that compositing changes (rendered via
    the CoreAnimation rendering display link) appear on screen at the same time
    as content painted into the window via the normal WebCore rendering path.

    CoreAnimation will commit any layer changes at the end of the runloop via
    its "CA commit" observer. Those changes can then appear onscreen at any time
    when the display link fires, which can result in unsynchronized rendering.

    To fix this, the GraphicsLayerCA code in WebCore does not change the CA
    layer tree during style changes and layout; it stores up all changes and
    commits them via flushCompositingState(). There are then two situations in
    which we can call flushCompositingState():

    1. When painting. LocalFrameView::paintContents() makes a call to flushCompositingState().

    2. When style changes/layout have made changes to the layer tree which do not
       result in painting. In this case we need a run loop observer to do a
       flushCompositingState() at an appropriate time. The observer will keep firing
       until the time is right (essentially when there are no more pending layouts).
*/

void WebViewRenderingUpdateScheduler::updateRendering()
{
    @autoreleasepool {
#if PLATFORM(MAC)
        NSWindow *window = [m_webView window];
#endif // PLATFORM(MAC)

        [m_webView _updateRendering];

#if PLATFORM(MAC)
        // AppKit may have disabled screen updates, thinking an upcoming window flush will re-enable them.
        // In case setNeedsDisplayInRect() has prevented the window from needing to be flushed, re-enable screen
        // updates here.
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        if (![window isFlushWindowDisabled])
            [window _enableScreenUpdatesIfNeeded];
ALLOW_DEPRECATED_DECLARATIONS_END
#endif
    }
}
