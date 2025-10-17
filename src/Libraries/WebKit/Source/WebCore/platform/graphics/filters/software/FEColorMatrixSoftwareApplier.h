/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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

#include "FilterEffectApplier.h"
#include <array>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FEColorMatrix;
class PixelBuffer;

class FEColorMatrixSoftwareApplier final : public FilterEffectConcreteApplier<FEColorMatrix> {
    WTF_MAKE_TZONE_ALLOCATED(FEColorMatrixSoftwareApplier);
    using Base = FilterEffectConcreteApplier<FEColorMatrix>;

public:
    FEColorMatrixSoftwareApplier(const FEColorMatrix&);

private:
    bool apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const final;

    inline void matrix(float& red, float& green, float& blue, float& alpha) const;
    inline void saturateAndHueRotate(float& red, float& green, float& blue) const;
    inline void luminance(float& red, float& green, float& blue, float& alpha) const;

#if USE(ACCELERATE)
    void applyPlatformAccelerated(PixelBuffer&) const;
#endif
    void applyPlatformUnaccelerated(PixelBuffer&) const;

    void applyPlatform(PixelBuffer&) const;

    std::array<float, 9> m_components;
};

} // namespace WebCore
