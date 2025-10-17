/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#import "SliderTrackPart.h"
#import <wtf/TZoneMalloc.h>

namespace WebCore {

class SliderTrackMac final : public ControlMac {
    WTF_MAKE_TZONE_ALLOCATED(SliderTrackMac);
public:
    SliderTrackMac(SliderTrackPart&, ControlFactoryMac&);

private:
    const SliderTrackPart& owningSliderTrackPart() const { return downcast<SliderTrackPart>(m_owningPart); }

    FloatRect rectForBounds(const FloatRect& bounds, const ControlStyle&) const override;

    void draw(GraphicsContext&, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle&) override;
};

} // namespace WebCore

#endif // PLATFORM(MAC)
