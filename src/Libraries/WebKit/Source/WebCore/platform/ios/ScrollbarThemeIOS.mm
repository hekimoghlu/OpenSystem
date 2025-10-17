/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#import "ScrollbarThemeIOS.h"

#if PLATFORM(IOS_FAMILY)

#import "GraphicsContext.h"
#import "IntRect.h"
#import "PlatformMouseEvent.h"
#import "Scrollbar.h"
#import <wtf/NeverDestroyed.h>
#import <wtf/StdLibExtras.h>

namespace WebCore {

ScrollbarTheme& ScrollbarTheme::nativeTheme()
{
    static NeverDestroyed<ScrollbarThemeIOS> theme;
    return theme;
}

void ScrollbarThemeIOS::registerScrollbar(Scrollbar&)
{
}

void ScrollbarThemeIOS::unregisterScrollbar(Scrollbar&)
{
}

ScrollbarThemeIOS::ScrollbarThemeIOS()
{
}

ScrollbarThemeIOS::~ScrollbarThemeIOS()
{
}

void ScrollbarThemeIOS::preferencesChanged()
{
}

int ScrollbarThemeIOS::scrollbarThickness(ScrollbarWidth, ScrollbarExpansionState, OverlayScrollbarSizeRelevancy)
{
    return 0;
}

Seconds ScrollbarThemeIOS::initialAutoscrollTimerDelay()
{
    return 0_s;
}

Seconds ScrollbarThemeIOS::autoscrollTimerDelay()
{
    return 0_s;
}
    
ScrollbarButtonsPlacement ScrollbarThemeIOS::buttonsPlacement() const
{
    return ScrollbarButtonsNone;
}

bool ScrollbarThemeIOS::hasButtons(Scrollbar&)
{
    return false;
}

bool ScrollbarThemeIOS::hasThumb(Scrollbar&)
{
    return false;
}

IntRect ScrollbarThemeIOS::backButtonRect(Scrollbar&, ScrollbarPart, bool /*painting*/)
{
    return IntRect();
}

IntRect ScrollbarThemeIOS::forwardButtonRect(Scrollbar&, ScrollbarPart, bool /*painting*/)
{
    return IntRect();
}

IntRect ScrollbarThemeIOS::trackRect(Scrollbar&, bool /*painting*/)
{
    return IntRect();
}

int ScrollbarThemeIOS::minimumThumbLength(Scrollbar&)
{
    return 0;
}

bool ScrollbarThemeIOS::paint(Scrollbar&, GraphicsContext&, const IntRect& /*damageRect*/)
{
    return true;
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
