/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
#include "config.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"

#include <limits>
#include <wtf/text/MakeString.h>

namespace WebCore {
namespace CSS {

static NEVER_INLINE ASCIILiteral formatNonfiniteCSSNumberValuePrefix(double number)
{
    if (number == std::numeric_limits<double>::infinity())
        return "infinity"_s;
    if (number == -std::numeric_limits<double>::infinity())
        return "-infinity"_s;
    ASSERT(std::isnan(number));
    return "NaN"_s;
}

NEVER_INLINE void formatNonfiniteCSSNumberValue(StringBuilder& builder, const SerializableNumber& number)
{
    return builder.append(formatNonfiniteCSSNumberValuePrefix(number.value), number.suffix.isEmpty() ? ""_s : " * 1"_s, number.suffix);
}

NEVER_INLINE String formatNonfiniteCSSNumberValue(const SerializableNumber& number)
{
    return makeString(formatNonfiniteCSSNumberValuePrefix(number.value), number.suffix.isEmpty() ? ""_s : " * 1"_s, number.suffix);
}

NEVER_INLINE void formatCSSNumberValue(StringBuilder& builder, const SerializableNumber& number)
{
    if (!std::isfinite(number.value))
        return formatNonfiniteCSSNumberValue(builder, number);
    return builder.append(FormattedCSSNumber::create(number.value), number.suffix);
}

NEVER_INLINE String formatCSSNumberValue(const SerializableNumber& number)
{
    if (!std::isfinite(number.value))
        return formatNonfiniteCSSNumberValue(number);
    return makeString(FormattedCSSNumber::create(number.value), number.suffix);
}

void Serialize<SerializableNumber>::operator()(StringBuilder& builder, const SerializableNumber& number)
{
    formatCSSNumberValue(builder, number);
}

} // namespace CSS
} // namespace WebCore
