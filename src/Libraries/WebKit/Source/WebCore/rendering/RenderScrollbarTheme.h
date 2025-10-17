/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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

#include "ScrollbarThemeComposite.h"

namespace WebCore {

class PlatformMouseEvent;
class Scrollbar;
class ScrollView;

class RenderScrollbarTheme final : public ScrollbarThemeComposite {
public:
    virtual ~RenderScrollbarTheme() = default;
    
    int scrollbarThickness(ScrollbarWidth scrollbarWidth = ScrollbarWidth::Auto, ScrollbarExpansionState expansionState = ScrollbarExpansionState::Expanded, OverlayScrollbarSizeRelevancy overlayRelevancy = OverlayScrollbarSizeRelevancy::IncludeOverlayScrollbarSize) override { return ScrollbarTheme::theme().scrollbarThickness(scrollbarWidth, expansionState, overlayRelevancy); }

    ScrollbarButtonsPlacement buttonsPlacement() const override { return ScrollbarTheme::theme().buttonsPlacement(); }

    bool supportsControlTints() const override { return true; }

    void paintScrollCorner(ScrollableArea&, GraphicsContext&, const IntRect& cornerRect) override;

    ScrollbarButtonPressAction handleMousePressEvent(Scrollbar& scrollbar, const PlatformMouseEvent& event, ScrollbarPart pressedPart) override { return ScrollbarTheme::theme().handleMousePressEvent(scrollbar, event, pressedPart); }

    Seconds initialAutoscrollTimerDelay() override { return ScrollbarTheme::theme().initialAutoscrollTimerDelay(); }
    Seconds autoscrollTimerDelay() override { return ScrollbarTheme::theme().autoscrollTimerDelay(); }

    void registerScrollbar(Scrollbar& scrollbar) override { return ScrollbarTheme::theme().registerScrollbar(scrollbar); }
    void unregisterScrollbar(Scrollbar& scrollbar) override { return ScrollbarTheme::theme().unregisterScrollbar(scrollbar); }

    int minimumThumbLength(Scrollbar&) override;

    void buttonSizesAlongTrackAxis(Scrollbar&, int& beforeSize, int& afterSize);
    
    static RenderScrollbarTheme* renderScrollbarTheme();

private:
    bool hasButtons(Scrollbar&) override;
    bool hasThumb(Scrollbar&) override;

    IntRect backButtonRect(Scrollbar&, ScrollbarPart, bool painting = false) override;
    IntRect forwardButtonRect(Scrollbar&, ScrollbarPart, bool painting = false) override;
    IntRect trackRect(Scrollbar&, bool painting = false) override;

    void willPaintScrollbar(GraphicsContext&, Scrollbar&) override;
    void didPaintScrollbar(GraphicsContext&, Scrollbar&) override;
    
    void paintScrollbarBackground(GraphicsContext&, Scrollbar&) override;
    void paintTrackBackground(GraphicsContext&, Scrollbar&, const IntRect&) override;
    void paintTrackPiece(GraphicsContext&, Scrollbar&, const IntRect&, ScrollbarPart) override;
    void paintButton(GraphicsContext&, Scrollbar&, const IntRect&, ScrollbarPart) override;
    void paintThumb(GraphicsContext&, Scrollbar&, const IntRect&) override;
    void paintTickmarks(GraphicsContext&, Scrollbar&, const IntRect&) override;

    IntRect constrainTrackRectToTrackPieces(Scrollbar&, const IntRect&) override;
};

} // namespace WebCore
