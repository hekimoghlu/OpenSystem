/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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

#if USE(THEME_ADWAITA)

#include "ScrollbarThemeComposite.h"

namespace WebCore {

class ScrollbarThemeAdwaita : public ScrollbarThemeComposite {
public:
    ScrollbarThemeAdwaita() = default;
    virtual ~ScrollbarThemeAdwaita() = default;

protected:
    bool usesOverlayScrollbars() const override;
    bool invalidateOnMouseEnterExit() override { return usesOverlayScrollbars(); }

    void updateScrollbarOverlayStyle(Scrollbar&) override;

    bool paint(Scrollbar&, GraphicsContext&, const IntRect&) override;
    void paintScrollCorner(ScrollableArea&, GraphicsContext&, const IntRect&) override;
    ScrollbarButtonPressAction handleMousePressEvent(Scrollbar&, const PlatformMouseEvent&, ScrollbarPart) override;

    int scrollbarThickness(ScrollbarWidth, ScrollbarExpansionState, OverlayScrollbarSizeRelevancy = OverlayScrollbarSizeRelevancy::IncludeOverlayScrollbarSize) override;
    int minimumThumbLength(Scrollbar&) override;

    bool hasButtons(Scrollbar&) override;
    bool hasThumb(Scrollbar&) override;

    IntRect backButtonRect(Scrollbar&, ScrollbarPart, bool painting = false) override;
    IntRect forwardButtonRect(Scrollbar&, ScrollbarPart, bool painting = false) override;
    IntRect trackRect(Scrollbar&, bool painting = false) override;
};

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
