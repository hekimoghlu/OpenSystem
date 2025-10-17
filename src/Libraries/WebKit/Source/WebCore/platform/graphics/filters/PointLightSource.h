/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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

#include "LightSource.h"
#include <wtf/Ref.h>

namespace WebCore {

class PointLightSource : public LightSource {
public:
    WEBCORE_EXPORT static Ref<PointLightSource> create(const FloatPoint3D& position);

    bool operator==(const PointLightSource&) const;

    const FloatPoint3D& position() const { return m_position; }
    bool setX(float) override;
    bool setY(float) override;
    bool setZ(float) override;

    void initPaintingData(const Filter&, const FilterImage& result, PaintingData&) const override;
    ComputedLightingData computePixelLightingData(const PaintingData&, int x, int y, float z) const final;

    WTF::TextStream& externalRepresentation(WTF::TextStream&) const override;

private:
    PointLightSource(const FloatPoint3D& position);

    bool operator==(const LightSource& other) const override { return areEqual<PointLightSource>(*this, other); }

    FloatPoint3D m_position;
    mutable FloatPoint3D m_bufferPosition;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_LIGHTSOURCE(PointLightSource, LightType::LS_POINT)
