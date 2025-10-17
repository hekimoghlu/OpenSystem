/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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

#import "ControlMac.h"
#import <wtf/TZoneMalloc.h>

namespace WebCore {

class InnerSpinButtonPart;

class InnerSpinButtonMac final : public ControlMac {
    WTF_MAKE_TZONE_ALLOCATED(InnerSpinButtonMac);
public:
    InnerSpinButtonMac(InnerSpinButtonPart&, ControlFactoryMac&, NSStepperCell *);

private:
    IntSize cellSize(NSControlSize, const ControlStyle&) const override;

    void draw(GraphicsContext&, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle&) override;
    void drawWithCoreUI(GraphicsContext&, const FloatRoundedRect& borderRect, const ControlStyle&);

#if HAVE(NSSTEPPERCELL_INCREMENTING)
    IntOutsets cellOutsets(NSControlSize, const ControlStyle&) const override;
    void drawWithCell(GraphicsContext&, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle&);
    FloatRect rectForBounds(const FloatRect&, const ControlStyle&) const override;
    void updateCellStates(const FloatRect&, const ControlStyle&) override;
#endif

    RetainPtr<NSStepperCell> m_stepperCell;
};

} // namespace WebCore

#endif // PLATFORM(MAC)
