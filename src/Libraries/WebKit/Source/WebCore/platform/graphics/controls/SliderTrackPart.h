/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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

#include "ControlPart.h"
#include "IntRect.h"

namespace WebCore {

class SliderTrackPart : public ControlPart {
public:
    static Ref<SliderTrackPart> create(StyleAppearance);
    WEBCORE_EXPORT static Ref<SliderTrackPart> create(StyleAppearance, const IntSize& thumbSize, const IntRect& trackBounds, Vector<double>&& tickRatios, double thumbPosition);

    IntSize thumbSize() const { return m_thumbSize; }
    void setThumbSize(IntSize thumbSize) { m_thumbSize = thumbSize; }

    IntRect trackBounds() const { return m_trackBounds; }
    void setTrackBounds(IntRect trackBounds) { m_trackBounds = trackBounds; }

    const Vector<double>& tickRatios() const { return m_tickRatios; }
    void setTickRatios(Vector<double>&& tickRatios) { m_tickRatios = WTFMove(tickRatios); }

    double thumbPosition() const { return m_thumbPosition; }
    void setThumbPosition(double thumbPosition) { m_thumbPosition = thumbPosition; }

    void drawTicks(GraphicsContext&, const FloatRect&, const ControlStyle&) const;

private:
    SliderTrackPart(StyleAppearance, const IntSize& thumbSize, const IntRect& trackBounds, Vector<double>&& tickRatios, double thumbPosition);

    std::unique_ptr<PlatformControl> createPlatformControl() override;

    IntSize m_thumbSize;
    IntRect m_trackBounds;
    Vector<double> m_tickRatios;
    double m_thumbPosition;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SliderTrackPart) \
    static bool isType(const WebCore::ControlPart& part) { return part.type() == WebCore::StyleAppearance::SliderHorizontal || part.type() == WebCore::StyleAppearance::SliderVertical; } \
SPECIALIZE_TYPE_TRAITS_END()
