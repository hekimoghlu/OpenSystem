/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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

#if PLATFORM(MAC)

#import "ScrollTypes.h"
#import <AppKit/AppKit.h>

namespace WebCore {

inline NSScrollerStyle nsScrollerStyle(ScrollbarStyle style)
{
    switch (style) {
    case ScrollbarStyle::AlwaysVisible:
        return NSScrollerStyleLegacy;
    case ScrollbarStyle::Overlay:
        return NSScrollerStyleOverlay;
    }
    ASSERT_NOT_REACHED();
    return NSScrollerStyleLegacy;
}

inline ScrollbarStyle scrollbarStyle(NSScrollerStyle style)
{
    switch (style) {
    case NSScrollerStyleLegacy:
        return ScrollbarStyle::AlwaysVisible;
    case NSScrollerStyleOverlay:
        return ScrollbarStyle::Overlay;
    }
    ASSERT_NOT_REACHED();
    return ScrollbarStyle::AlwaysVisible;
}

inline NSControlSize nsControlSizeFromScrollbarWidth(ScrollbarWidth width)
{
    switch (width) {
    case ScrollbarWidth::Auto:
    case ScrollbarWidth::None:
        return NSControlSizeRegular;
    case ScrollbarWidth::Thin:
        return NSControlSizeSmall;
    }

    ASSERT_NOT_REACHED();
    return NSControlSizeRegular;
}

} // namespace WebCore

#endif // PLATFORM(MAC)
