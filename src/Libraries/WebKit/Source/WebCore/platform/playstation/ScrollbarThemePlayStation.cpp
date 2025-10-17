/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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
#include "config.h"
#include "ScrollbarThemePlayStation.h"

#include "HostWindow.h"
#include "NotImplemented.h"
#include "ScrollView.h"
#include "Scrollbar.h"

namespace WebCore {

ScrollbarTheme& ScrollbarTheme::nativeTheme()
{
    static ScrollbarThemePlayStation theme;
    return theme;
}

int ScrollbarThemePlayStation::scrollbarThickness(ScrollbarWidth scrollbarWidth, ScrollbarExpansionState, OverlayScrollbarSizeRelevancy)
{
    if (scrollbarWidth == ScrollbarWidth::None)
        return 0;
    return 7;
}

bool ScrollbarThemePlayStation::hasButtons(Scrollbar&)
{
    notImplemented();
    return true;
}

bool ScrollbarThemePlayStation::hasThumb(Scrollbar&)
{
    notImplemented();
    return true;
}

IntRect ScrollbarThemePlayStation::backButtonRect(Scrollbar&, ScrollbarPart, bool)
{
    notImplemented();
    return { };
}

IntRect ScrollbarThemePlayStation::forwardButtonRect(Scrollbar&, ScrollbarPart, bool)
{
    notImplemented();
    return { };
}

IntRect ScrollbarThemePlayStation::trackRect(Scrollbar& scrollbar, bool)
{
    return scrollbar.frameRect();
}

void ScrollbarThemePlayStation::paintTrackBackground(GraphicsContext& context, Scrollbar& scrollbar, const IntRect& trackRect)
{
    context.fillRect(trackRect, scrollbar.enabled() ? Color::lightGray : SRGBA<uint8_t> { 224, 224, 224 });
}

void ScrollbarThemePlayStation::paintThumb(GraphicsContext& context, Scrollbar& scrollbar, const IntRect& thumbRect)
{
    if (scrollbar.enabled())
        context.fillRect(thumbRect, Color::darkGray);
}

} // namespace WebCore
