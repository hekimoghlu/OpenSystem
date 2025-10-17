/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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
#import "MeterPart.h"
#import <wtf/TZoneMalloc.h>

namespace WebCore {

class MeterMac final : public ControlMac {
    WTF_MAKE_TZONE_ALLOCATED(MeterMac);
public:
    MeterMac(MeterPart& owningMeterPart, ControlFactoryMac&, NSLevelIndicatorCell*);

private:
    const MeterPart& owningMeterPart() const { return downcast<MeterPart>(m_owningPart); }

    void updateCellStates(const FloatRect&, const ControlStyle&) override;

    FloatSize sizeForBounds(const FloatRect& bounds, const ControlStyle&) const override;

    void draw(GraphicsContext&, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle&) override;

    RetainPtr<NSLevelIndicatorCell> m_levelIndicatorCell;
};

} // namespace WebCore

#endif // PLATFORM(MAC)
