/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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

#include "InputMethodState.h"
#include <WebCore/CompositionUnderline.h>
#include <WebCore/IntPoint.h>
#include <wtf/Noncopyable.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>

typedef struct _WebKitInputMethodContext WebKitInputMethodContext;

#if PLATFORM(GTK)
#if USE(GTK4)
typedef struct _GdkEvent GdkEvent;
#else
typedef union _GdkEvent GdkEvent;
#endif
#elif PLATFORM(WPE)
struct wpe_input_keyboard_event;
#endif

namespace WebCore {
class IntRect;
}

namespace WebKit {

class InputMethodFilter {
    WTF_MAKE_NONCOPYABLE(InputMethodFilter);
public:

    InputMethodFilter() = default;
    ~InputMethodFilter();

    void setContext(WebKitInputMethodContext*);
    WebKitInputMethodContext* context() const { return m_context.get(); }

    void setState(std::optional<InputMethodState>&&);

#if PLATFORM(GTK)
    using PlatformEventKey = GdkEvent;
#elif PLATFORM(WPE)
    using PlatformEventKey = void;

#if ENABLE(WPE_PLATFORM)
    void setUseWPEPlatformEvents(bool useWPEPlatformEvents) { m_useWPEPlatformEvents = useWPEPlatformEvents; }
#endif
#endif
    struct FilterResult {
        bool handled { false };
        String keyText;
    };
    FilterResult filterKeyEvent(PlatformEventKey*);

#if PLATFORM(GTK)
    FilterResult filterKeyEvent(unsigned type, unsigned keyval, unsigned keycode, unsigned modifiers);
#endif

    void notifyFocusedIn();
    void notifyFocusedOut();
    void notifyMouseButtonPress();
    void notifyCursorRect(const WebCore::IntRect&);
    void notifySurrounding(const String&, uint64_t, uint64_t);

    void cancelComposition();

private:
    static void preeditStartedCallback(InputMethodFilter*);
    static void preeditChangedCallback(InputMethodFilter*);
    static void preeditFinishedCallback(InputMethodFilter*);
    static void committedCallback(InputMethodFilter*, const char*);
    static void deleteSurroundingCallback(InputMethodFilter*, int offset, unsigned characterCount);

    void preeditStarted();
    void preeditChanged();
    void preeditFinished();
    void committed(const char*);
    void deleteSurrounding(int offset, unsigned characterCount);

    bool isEnabled() const { return !!m_state; }
    bool isViewFocused() const;

    void notifyContentType();

    WebCore::IntRect platformTransformCursorRectToViewCoordinates(const WebCore::IntRect&);
    bool platformEventKeyIsKeyPress(PlatformEventKey*) const;

    std::optional<InputMethodState> m_state;
    GRefPtr<WebKitInputMethodContext> m_context;

    struct {
        String text;
        Vector<WebCore::CompositionUnderline> underlines;
        unsigned cursorOffset;
    } m_preedit;

    struct {
        bool isActive { false };
        bool preeditChanged { false };
#if PLATFORM(GTK) && USE(GTK4)
        bool isFakeKeyEventForTesting { false };
#endif
    } m_filteringContext;

    String m_compositionResult;
    WebCore::IntPoint m_cursorLocation;

    struct {
        String text;
        uint64_t cursorPosition;
        uint64_t selectionPosition;
    } m_surrounding;

#if ENABLE(WPE_PLATFORM)
    bool m_useWPEPlatformEvents { false };
#endif
};

} // namespace WebKit
