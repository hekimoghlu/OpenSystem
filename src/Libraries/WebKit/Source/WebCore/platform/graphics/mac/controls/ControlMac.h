/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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

#import "LengthBox.h"
#import "PlatformControl.h"
#import <wtf/TZoneMalloc.h>

namespace WebCore {

class ControlFactoryMac;
class FloatSize;
class GraphicsContext;
class IntSize;

class ControlMac : public PlatformControl {
    WTF_MAKE_TZONE_ALLOCATED(ControlMac);
public:
    ControlMac(ControlPart&, ControlFactoryMac&);

protected:
    static bool userPrefersContrast();
    static bool userPrefersWithoutColorDifferentiation();
    static FloatRect inflatedRect(const FloatRect& bounds, const FloatSize&, const IntOutsets&, const ControlStyle&);

    static void updateCheckedState(NSCell *, const ControlStyle&);
    static void updateEnabledState(NSCell *, const ControlStyle&);
    static void updateFocusedState(NSCell *, const ControlStyle&);
    static void updatePressedState(NSCell *, const ControlStyle&);

    virtual IntSize cellSize(NSControlSize, const ControlStyle&) const { return { }; };
    virtual IntOutsets cellOutsets(NSControlSize, const ControlStyle&) const { return { }; };

    NSControlSize controlSizeForFont(const ControlStyle&) const;
    NSControlSize controlSizeForSystemFont(const ControlStyle&) const;
    NSControlSize controlSizeForSize(const FloatSize&, const ControlStyle&) const;

    IntSize sizeForSystemFont(const ControlStyle&) const;

    void setFocusRingClipRect(const FloatRect& clipBounds) override;

    void updateCellStates(const FloatRect&, const ControlStyle&) override;

    void drawCell(GraphicsContext&, const FloatRect&, float deviceScaleFactor, const ControlStyle&, NSCell *, bool drawCell = true);

    void drawListButton(GraphicsContext&, const FloatRect&, float deviceScaleFactor, const ControlStyle&);

private:
    void drawCellInternal(GraphicsContext&, const FloatRect&, float deviceScaleFactor, const ControlStyle&, NSCell *);
    void drawCellFocusRingInternal(GraphicsContext&, const FloatRect&, float deviceScaleFactor, const ControlStyle&, NSCell *);

    void drawCellFocusRing(GraphicsContext&, const FloatRect&, float deviceScaleFactor, const ControlStyle&, NSCell *);
    void drawCellOrFocusRing(GraphicsContext&, const FloatRect&, float deviceScaleFactor, const ControlStyle&, NSCell *, bool drawCell = true);

    ControlFactoryMac& m_controlFactory;
};

} // namespace WebCore

#endif // PLATFORM(MAC)
