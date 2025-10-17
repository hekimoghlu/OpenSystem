/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 1, 2024.
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

#include "ScrollbarTheme.h"

#if PLATFORM(COCOA)
OBJC_CLASS NSScrollerImp;
#endif

namespace WebCore {

class ScrollbarThemeComposite : public ScrollbarTheme {
public:
    // Implement ScrollbarTheme interface
    bool paint(Scrollbar&, GraphicsContext&, const IntRect& damageRect) override;
    ScrollbarPart hitTest(Scrollbar&, const IntPoint&) override;
    void invalidatePart(Scrollbar&, ScrollbarPart) override;
    int thumbPosition(Scrollbar&) override;
    int thumbLength(Scrollbar&) override;
    int trackPosition(Scrollbar&) override;
    int trackLength(Scrollbar&) override;
    void paintOverhangAreas(ScrollView&, GraphicsContext&, const IntRect& horizontalOverhangArea, const IntRect& verticalOverhangArea, const IntRect& dirtyRect) override;

    virtual bool hasButtons(Scrollbar&) = 0;
    virtual bool hasThumb(Scrollbar&) = 0;

    virtual IntRect backButtonRect(Scrollbar&, ScrollbarPart, bool painting = false) = 0;
    virtual IntRect forwardButtonRect(Scrollbar&, ScrollbarPart, bool painting = false) = 0;
    virtual IntRect trackRect(Scrollbar&, bool painting = false) = 0;
    virtual IntRect thumbRect(Scrollbar&);

    virtual void splitTrack(Scrollbar&, const IntRect& track, IntRect& startTrack, IntRect& thumb, IntRect& endTrack);
    
    virtual int minimumThumbLength(Scrollbar&);

    virtual void willPaintScrollbar(GraphicsContext&, Scrollbar&) { }
    virtual void didPaintScrollbar(GraphicsContext&, Scrollbar&) { }

    virtual void paintScrollbarBackground(GraphicsContext&, Scrollbar&) { }
    virtual void paintTrackBackground(GraphicsContext&, Scrollbar&, const IntRect&) { }
    virtual void paintTrackPiece(GraphicsContext&, Scrollbar&, const IntRect&, ScrollbarPart) { }
    virtual void paintButton(GraphicsContext&, Scrollbar&, const IntRect&, ScrollbarPart) { }
    virtual void paintThumb(GraphicsContext&, Scrollbar&, const IntRect&) { }

    virtual IntRect constrainTrackRectToTrackPieces(Scrollbar&, const IntRect& rect) { return rect; }
};

}
