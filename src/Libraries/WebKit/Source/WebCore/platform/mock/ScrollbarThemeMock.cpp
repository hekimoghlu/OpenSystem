/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#include "ScrollbarThemeMock.h"

// FIXME: This is a layering violation.
#include "DeprecatedGlobalSettings.h"
#include "Scrollbar.h"

namespace WebCore {

IntRect ScrollbarThemeMock::trackRect(Scrollbar& scrollbar, bool)
{
    return scrollbar.frameRect();
}

int ScrollbarThemeMock::scrollbarThickness(ScrollbarWidth scrollbarWidth, ScrollbarExpansionState, OverlayScrollbarSizeRelevancy overlayRelavancy)
{
    if (usesOverlayScrollbars() && overlayRelavancy == OverlayScrollbarSizeRelevancy::IgnoreOverlayScrollbarSize)
        return 0;

    switch (scrollbarWidth) {
    case ScrollbarWidth::Auto:
        return 15;
    case ScrollbarWidth::Thin:
        return 11;
    case ScrollbarWidth::None:
        return 0;
    }
    ASSERT_NOT_REACHED();
    return 15;
}

void ScrollbarThemeMock::paintTrackBackground(GraphicsContext& context, Scrollbar& scrollbar, const IntRect& trackRect)
{
    context.fillRect(trackRect, scrollbar.enabled() ? Color::lightGray : SRGBA<uint8_t> { 224, 224, 224 });
}

void ScrollbarThemeMock::paintThumb(GraphicsContext& context, Scrollbar& scrollbar, const IntRect& thumbRect)
{
    if (scrollbar.enabled())
        context.fillRect(thumbRect, Color::darkGray);
}

bool ScrollbarThemeMock::usesOverlayScrollbars() const
{
    // FIXME: This is a layering violation, but ScrollbarThemeMock is also created depending on settings in platform layer,
    // we should fix it in both places.
    return DeprecatedGlobalSettings::usesOverlayScrollbars();
}

}

