/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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

#if HAVE(ARM_NEON_INTRINSICS)

#include "FilterEffectApplier.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FEComposite;

class FECompositeNeonArithmeticApplier final : public FilterEffectConcreteApplier<FEComposite> {
    WTF_MAKE_TZONE_ALLOCATED(FECompositeNeonArithmeticApplier);
    using Base = FilterEffectConcreteApplier<FEComposite>;

public:
    FECompositeNeonArithmeticApplier(const FEComposite&);

private:
    template <int b1, int b4>
    static inline void computePixels(const uint8_t* source, uint8_t* destination, unsigned pixelArrayLength, float k1, float k2, float k3, float k4);

    static inline void applyPlatform(const uint8_t* source, uint8_t* destination, unsigned pixelArrayLength, float k1, float k2, float k3, float k4);

    bool apply(const Filter&, const FilterImageVector& inputs, FilterImage& result) const final;
};

} // namespace WebCore

#endif // HAVE(ARM_NEON_INTRINSICS)
