/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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

#include "WPEWebView.h"
#include <wtf/glib/GRefPtr.h>

struct wpe_input_keyboard_event;

namespace WebKit {
class TouchGestureController;
}

namespace WKWPE {

class ViewLegacy final : public View {
public:
    static Ref<View> create(struct wpe_view_backend* backend, const API::PageConfiguration& configuration)
    {
        return adoptRef(*new ViewLegacy(backend, configuration));
    }
    ~ViewLegacy();

#if ENABLE(FULLSCREEN_API)
    bool setFullScreen(bool);
#endif

#if ENABLE(TOUCH_EVENTS)
    WebKit::TouchGestureController& touchGestureController() const { return *m_touchGestureController; }
#endif

#if ENABLE(GAMEPAD)
    static WebKit::WebPageProxy* platformWebPageProxyForGamepadInput();
#endif

private:
    ViewLegacy(struct wpe_view_backend*, const API::PageConfiguration&);

    struct wpe_view_backend* backend() const override { return m_backend; }
    void synthesizeCompositionKeyPress(const String&, std::optional<Vector<WebCore::CompositionUnderline>>&&, std::optional<WebKit::EditingRange>&&) override;
    void callAfterNextPresentationUpdate(CompletionHandler<void()>&&) override;

    void setViewState(OptionSet<WebCore::ActivityState>);
    void handleKeyboardEvent(struct wpe_input_keyboard_event*);

    struct wpe_view_backend* m_backend { nullptr };
    bool m_horizontalScrollActive { false };
    bool m_verticalScrollActive { false };
#if ENABLE(TOUCH_EVENTS)
    std::unique_ptr<WebKit::TouchGestureController> m_touchGestureController;
#endif
};

} // namespace WKWPE
