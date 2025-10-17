/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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

#if PLATFORM(IOS_FAMILY)

#import "WKBrowserEngineDefinitions.h"
#import "WKWebView.h"
#import <WebCore/FloatPoint.h>
#import <WebCore/FloatSize.h>
#import <wtf/RunLoop.h>
#import <wtf/WeakObjCPtr.h>

@class BEScrollViewScrollUpdate;
@class UIScrollEvent;
@class UIWindow;
@class WKBaseScrollView;

namespace WebKit {
class PointerTouchCompatibilitySimulator;
}

namespace WTF {
template<typename T> struct IsDeprecatedTimerSmartPointerException;
template<> struct IsDeprecatedTimerSmartPointerException<WebKit::PointerTouchCompatibilitySimulator> : std::true_type { };
}

namespace WebKit {

class PointerTouchCompatibilitySimulator {
    WTF_MAKE_NONCOPYABLE(PointerTouchCompatibilitySimulator);
    WTF_MAKE_TZONE_ALLOCATED(PointerTouchCompatibilitySimulator);
public:
    PointerTouchCompatibilitySimulator(WKWebView *);

#if HAVE(UISCROLLVIEW_ASYNCHRONOUS_SCROLL_EVENT_HANDLING)
    bool handleScrollUpdate(WKBaseScrollView *, WKBEScrollViewScrollUpdate *);
#endif

    bool isSimulatingTouches() const { return !m_touchDelta.isZero(); }
    void setEnabled(bool);

    RetainPtr<WKWebView> view() const { return m_view.get(); }
    RetainPtr<UIWindow> window() const;

private:
    void resetState();
    WebCore::FloatPoint locationInScreen() const;

    const WeakObjCPtr<WKWebView> m_view;
    RunLoop::Timer m_stateResetWatchdogTimer;
    WebCore::FloatPoint m_centroid;
    WebCore::FloatSize m_touchDelta;
    WebCore::FloatSize m_initialDelta;
    bool m_isEnabled { false };
};

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
