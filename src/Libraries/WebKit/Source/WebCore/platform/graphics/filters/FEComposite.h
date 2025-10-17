/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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

#include "FilterEffect.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class CompositeOperationType : uint8_t {
    FECOMPOSITE_OPERATOR_UNKNOWN    = 0,
    FECOMPOSITE_OPERATOR_OVER       = 1,
    FECOMPOSITE_OPERATOR_IN         = 2,
    FECOMPOSITE_OPERATOR_OUT        = 3,
    FECOMPOSITE_OPERATOR_ATOP       = 4,
    FECOMPOSITE_OPERATOR_XOR        = 5,
    FECOMPOSITE_OPERATOR_ARITHMETIC = 6,
    FECOMPOSITE_OPERATOR_LIGHTER    = 7
};

class FEComposite : public FilterEffect {
public:
    WEBCORE_EXPORT static Ref<FEComposite> create(const CompositeOperationType&, float k1, float k2, float k3, float k4, DestinationColorSpace = DestinationColorSpace::SRGB());

    bool operator==(const FEComposite&) const;

    CompositeOperationType operation() const { return m_type; }
    bool setOperation(CompositeOperationType);

    float k1() const { return m_k1; }
    bool setK1(float);

    float k2() const { return m_k2; }
    bool setK2(float);

    float k3() const { return m_k3; }
    bool setK3(float);

    float k4() const { return m_k4; }
    bool setK4(float);

private:
    FEComposite(const CompositeOperationType&, float k1, float k2, float k3, float k4, DestinationColorSpace);

    bool operator==(const FilterEffect& other) const override { return areEqual<FEComposite>(*this, other); }

    unsigned numberOfEffectInputs() const override { return 2; }

    FloatRect calculateImageRect(const Filter&, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const override;

    bool resultIsValidPremultiplied() const override { return m_type != CompositeOperationType::FECOMPOSITE_OPERATOR_ARITHMETIC; }

    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const override;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

#if HAVE(ARM_NEON_INTRINSICS)
    template <int b1, int b4>
    static inline void computeArithmeticPixelsNeon(const uint8_t* source, uint8_t* destination, unsigned pixelArrayLength, float k1, float k2, float k3, float k4);

    static inline void platformArithmeticNeon(const uint8_t* source, uint8_t* destination, unsigned pixelArrayLength, float k1, float k2, float k3, float k4);
#endif

    CompositeOperationType m_type;
    float m_k1;
    float m_k2;
    float m_k3;
    float m_k4;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FEComposite)
