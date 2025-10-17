/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
#ifndef ScrollbarThemeMock_h
#define ScrollbarThemeMock_h

#include "ScrollbarThemeComposite.h"

namespace WebCore {

// Scrollbar theme used in image snapshots, to eliminate appearance differences between platforms.
class ScrollbarThemeMock : public ScrollbarThemeComposite {
public:
    int scrollbarThickness(ScrollbarWidth = ScrollbarWidth::Auto, ScrollbarExpansionState = ScrollbarExpansionState::Expanded, OverlayScrollbarSizeRelevancy = OverlayScrollbarSizeRelevancy::IncludeOverlayScrollbarSize) override;

protected:
    bool hasButtons(Scrollbar&) override { return false; }
    bool hasThumb(Scrollbar&) override  { return true; }

    IntRect backButtonRect(Scrollbar&, ScrollbarPart, bool /*painting*/ = false) override { return IntRect(); }
    IntRect forwardButtonRect(Scrollbar&, ScrollbarPart, bool /*painting*/ = false) override { return IntRect(); }
    IntRect trackRect(Scrollbar&, bool painting = false) override;
    
    void paintTrackBackground(GraphicsContext&, Scrollbar&, const IntRect&) override;
    void paintThumb(GraphicsContext&, Scrollbar&, const IntRect&) override;
    int maxOverlapBetweenPages() override { return 40; }

    bool usesOverlayScrollbars() const override;
private:
    bool isMockTheme() const override { return true; }
};

}
#endif // ScrollbarThemeMock_h
