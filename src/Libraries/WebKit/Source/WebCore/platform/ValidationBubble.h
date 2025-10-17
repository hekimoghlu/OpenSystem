/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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

#include "IntRect.h"
#include <wtf/Forward.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
#include <wtf/RetainPtr.h>
#include <wtf/WeakObjCPtr.h>
#endif

#if PLATFORM(MAC)
OBJC_CLASS NSPopover;
#elif PLATFORM(IOS_FAMILY)
OBJC_CLASS UIViewController;
OBJC_CLASS WebValidationBubbleDelegate;
OBJC_CLASS WebValidationBubbleTapRecognizer;
OBJC_CLASS WebValidationBubbleViewController;
#endif

#if PLATFORM(MAC)
OBJC_CLASS NSView;
using PlatformView = NSView;
#elif PLATFORM(IOS_FAMILY)
OBJC_CLASS UIView;
using PlatformView = UIView;
#elif PLATFORM(GTK)
using PlatformView = GtkWidget;
#else
using PlatformView = void;
#endif

namespace WebCore {

class ValidationBubble : public RefCounted<ValidationBubble> {
public:
    struct Settings {
        double minimumFontSize { 0 };
    };

#if PLATFORM(GTK)
    using ShouldNotifyFocusEventsCallback = Function<void(PlatformView*, bool shouldNotifyFocusEvents)>;
    static Ref<ValidationBubble> create(PlatformView* view, const String& message, const Settings& settings, ShouldNotifyFocusEventsCallback&& callback)
    {
        return adoptRef(*new ValidationBubble(view, message, settings, WTFMove(callback)));
    }
#else
    static Ref<ValidationBubble> create(PlatformView* view, const String& message, const Settings& settings)
    {
        return adoptRef(*new ValidationBubble(view, message, settings));
    }
#endif

    WEBCORE_EXPORT ~ValidationBubble();

    const String& message() const { return m_message; }
    double fontSize() const { return m_fontSize; }

#if PLATFORM(IOS_FAMILY)
    WEBCORE_EXPORT void setAnchorRect(const IntRect& anchorRect, UIViewController* presentingViewController = nullptr);
    WEBCORE_EXPORT void show();
#else
    WEBCORE_EXPORT void showRelativeTo(const IntRect& anchorRect);
#endif

private:
#if PLATFORM(GTK)
    WEBCORE_EXPORT ValidationBubble(PlatformView*, const String& message, const Settings&, ShouldNotifyFocusEventsCallback&&);
    void invalidate();
#else
    WEBCORE_EXPORT ValidationBubble(PlatformView*, const String& message, const Settings&);
#endif

    PlatformView* m_view;
    String m_message;
    double m_fontSize { 0 };
#if PLATFORM(MAC)
    RetainPtr<NSPopover> m_popover;
#elif PLATFORM(IOS_FAMILY)
    RetainPtr<WebValidationBubbleViewController> m_popoverController;
    RetainPtr<WebValidationBubbleTapRecognizer> m_tapRecognizer;
    RetainPtr<WebValidationBubbleDelegate> m_popoverDelegate;
    WeakObjCPtr<UIViewController> m_presentingViewController;
    bool m_startingToPresentViewController { false };
#elif PLATFORM(GTK)
    GtkWidget* m_popover { nullptr };
    ShouldNotifyFocusEventsCallback m_shouldNotifyFocusEventsCallback { nullptr };
#endif
};

}
