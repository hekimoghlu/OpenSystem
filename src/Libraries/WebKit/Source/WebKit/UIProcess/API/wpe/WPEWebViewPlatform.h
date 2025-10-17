/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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

#if ENABLE(WPE_PLATFORM)

#include "GRefPtrWPE.h"
#include "RendererBufferFormat.h"
#include "WPEWebView.h"
#include <wpe/wpe-platform.h>
#include <wtf/HashMap.h>
#include <wtf/glib/GRefPtr.h>

namespace WebKit {
class AcceleratedBackingStoreDMABuf;
class WebPlatformTouchPoint;
}

namespace WKWPE {

class ViewPlatform final : public View {
public:
    static Ref<View> create(WPEDisplay* display, const API::PageConfiguration& configuration)
    {
        return adoptRef(*new ViewPlatform(display, configuration));
    }
    ~ViewPlatform();

#if ENABLE(FULLSCREEN_API)
    void enterFullScreen();
    void didEnterFullScreen();
    void exitFullScreen();
    void didExitFullScreen();
    void requestExitFullScreen();
#endif

    void updateAcceleratedSurface(uint64_t);
    WebKit::RendererBufferFormat renderBufferFormat() const;

private:
    ViewPlatform(WPEDisplay*, const API::PageConfiguration&);

    WPEView* wpeView() const override { return m_wpeView.get(); }
    void synthesizeCompositionKeyPress(const String&, std::optional<Vector<WebCore::CompositionUnderline>>&&, std::optional<WebKit::EditingRange>&&) override;
    void callAfterNextPresentationUpdate(CompletionHandler<void()>&&) override;
    void setCursor(const WebCore::Cursor&) override;

    void updateDisplayID();
    bool activityStateChanged(WebCore::ActivityState, bool);
    void toplevelStateChanged(WPEToplevelState previousState, WPEToplevelState);

#if ENABLE(POINTER_LOCK)
    void requestPointerLock() override;
    void didLosePointerLock() override;
#endif

#if ENABLE(TOUCH_EVENTS)
    Vector<WebKit::WebPlatformTouchPoint> touchPointsForEvent(WPEEvent*);
#endif

    gboolean handleEvent(WPEEvent*);
    void handleGesture(WPEEvent*);

    GRefPtr<WPEView> m_wpeView;
    RefPtr<WebKit::AcceleratedBackingStoreDMABuf> m_backingStore;
    uint32_t m_displayID { 0 };
    unsigned long m_bufferRenderedID { 0 };
    CompletionHandler<void()> m_nextPresentationUpdateCallback;
    HashMap<uint32_t, GRefPtr<WPEEvent>, IntHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>> m_touchEvents;
#if ENABLE(FULLSCREEN_API)
    bool m_viewWasAlreadyInFullScreen { false };
#endif
};

} // namespace WKWPE

#endif // ENABLE(WPE_PLATFORM)
