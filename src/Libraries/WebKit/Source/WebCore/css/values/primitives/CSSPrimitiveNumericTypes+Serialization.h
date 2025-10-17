/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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

#include "CSSPrimitiveNumericTypes.h"

namespace WebCore {
namespace CSS {

// MARK: - Serialization

struct SerializableNumber {
    double value;
    ASCIILiteral suffix;
};

void formatNonfiniteCSSNumberValue(StringBuilder&, const SerializableNumber&);
String formatNonfiniteCSSNumberValue(const SerializableNumber&);

void formatCSSNumberValue(StringBuilder&, const SerializableNumber&);
String formatCSSNumberValue(const SerializableNumber&);

template<> struct Serialize<SerializableNumber> {
    void operator()(StringBuilder&, const SerializableNumber&);
};

template<NumericRaw RawType> struct Serialize<RawType> {
    void operator()(StringBuilder& builder, const RawType& value)
    {
        serializationForCSS(builder, SerializableNumber { value.value, unitString(value.unit) });
    }
};

template<auto nR, auto pR, typename V> struct Serialize<NumberOrPercentageResolvedToNumber<nR, pR, V>> {
    void operator()(StringBuilder& builder, const NumberOrPercentageResolvedToNumber<nR, pR, V>& value)
    {
        WTF::switchOn(value,
            [&](const typename NumberOrPercentageResolvedToNumber<nR, pR, V>::Number& number) {
                serializationForCSS(builder, number);
            },
            [&](const typename NumberOrPercentageResolvedToNumber<nR, pR, V>::Percentage& percentage) {
                if (auto raw = percentage.raw())
                    serializationForCSS(builder, NumberRaw<nR, V> { raw->value / 100.0 });
                else
                    serializationForCSS(builder, percentage);
            }
        );
    }
};

} // namespace CSS
} // namespace WebCore
