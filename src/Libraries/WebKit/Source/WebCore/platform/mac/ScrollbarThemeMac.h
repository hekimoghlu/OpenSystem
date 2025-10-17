/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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

#if PLATFORM(MAC)

OBJC_CLASS CALayer;

namespace WebCore {

class ScrollbarThemeMac : public ScrollbarThemeComposite {
public:
    ScrollbarThemeMac();
    virtual ~ScrollbarThemeMac();

    WEBCORE_EXPORT void preferencesChanged();

    void updateEnabledState(Scrollbar&) override;

    bool paint(Scrollbar&, GraphicsContext&, const IntRect& damageRect) override;
    void paintScrollCorner(ScrollableArea&, GraphicsContext&, const IntRect& cornerRect) override;

    int scrollbarThickness(ScrollbarWidth = ScrollbarWidth::Auto, ScrollbarExpansionState = ScrollbarExpansionState::Expanded, OverlayScrollbarSizeRelevancy = OverlayScrollbarSizeRelevancy::IncludeOverlayScrollbarSize) override;
    
    bool supportsControlTints() const override { return true; }
    bool usesOverlayScrollbars() const  override;
    void usesOverlayScrollbarsChanged() override;
    void updateScrollbarOverlayStyle(Scrollbar&)  override;

    Seconds initialAutoscrollTimerDelay() override { return 500_ms; }
    Seconds autoscrollTimerDelay() override { return 50_ms; }

    ScrollbarButtonsPlacement buttonsPlacement() const override;

    void registerScrollbar(Scrollbar&) override;
    void unregisterScrollbar(Scrollbar&) override;

    static NSScrollerImp *scrollerImpForScrollbar(Scrollbar&);

    void setPaintCharacteristicsForScrollbar(Scrollbar&);

    static bool isCurrentlyDrawingIntoLayer();
    static void setIsCurrentlyDrawingIntoLayer(bool);

    void didCreateScrollerImp(Scrollbar&) override;
    bool isLayoutDirectionRTL(Scrollbar&);

#if HAVE(RUBBER_BANDING)
    WEBCORE_EXPORT static void setUpOverhangAreaShadow(CALayer *);
    WEBCORE_EXPORT static void removeOverhangAreaShadow(CALayer *);
#endif

protected:
    bool hasButtons(Scrollbar&) override;
    bool hasThumb(Scrollbar&) override;

    IntRect backButtonRect(Scrollbar&, ScrollbarPart, bool painting = false) override;
    IntRect forwardButtonRect(Scrollbar&, ScrollbarPart, bool painting = false) override;
    IntRect trackRect(Scrollbar&, bool painting = false) override;

    int maxOverlapBetweenPages() override { return 40; }

    int minimumThumbLength(Scrollbar&) override;
    
    ScrollbarButtonPressAction handleMousePressEvent(Scrollbar&, const PlatformMouseEvent&, ScrollbarPart) override;
    bool shouldDragDocumentInsteadOfThumb(Scrollbar&, const PlatformMouseEvent&) override;
    int scrollbarPartToHIPressedState(ScrollbarPart);
};

}

#endif // PLATFORM(MAC)
