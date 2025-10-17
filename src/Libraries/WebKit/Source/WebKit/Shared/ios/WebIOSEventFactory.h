/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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
#import "WebKeyboardEvent.h"
#import "WebMouseEvent.h"
#import "WebWheelEvent.h"
#import <UIKit/UIKit.h>
#import <WebCore/FloatSize.h>
#import <WebCore/WebEvent.h>

OBJC_CLASS WKBEScrollViewScrollUpdate;

namespace WebKit {

class WebIOSEventFactory {
public:
    static WebKit::WebKeyboardEvent createWebKeyboardEvent(::WebEvent *, bool handledByInputMethod);
    static WebKit::WebMouseEvent createWebMouseEvent(::WebEvent *);

#if HAVE(UISCROLLVIEW_ASYNCHRONOUS_SCROLL_EVENT_HANDLING)
    static WebKit::WebWheelEvent createWebWheelEvent(WKBEScrollViewScrollUpdate *, UIView *contentView, std::optional<WebKit::WebWheelEvent::Phase> overridePhase = std::nullopt);
    static WebCore::FloatSize translationInView(WKBEScrollViewScrollUpdate *, UIView *);
#endif

    static OptionSet<WebKit::WebEventModifier> webEventModifiersForUIKeyModifierFlags(UIKeyModifierFlags);
    static UIKeyModifierFlags toUIKeyModifierFlags(OptionSet<WebKit::WebEventModifier>);
    static UIEventButtonMask toUIEventButtonMask(WebKit::WebMouseEventButton);
};

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
