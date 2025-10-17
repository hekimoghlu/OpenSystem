/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
#include "CSSBorderRadius.h"

#include "CSSPrimitiveNumericTypes+Serialization.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace CSS {

static bool hasDefaultValueForAxis(const SpaceSeparatedArray<LengthPercentage<Nonnegative>, 4>& values)
{
    return values.value[3] == values.value[1]
        && values.value[2] == values.value[0]
        && values.value[1] == values.value[0]
        && values.value[0] == 0_css_px;
}

bool hasDefaultValue(const BorderRadius& borderRadius)
{
    return hasDefaultValueForAxis(borderRadius.horizontal) && hasDefaultValueForAxis(borderRadius.vertical);
}

static std::pair<SpaceSeparatedVector<LengthPercentage<Nonnegative>, 4>, bool> gatherSerializableRadiiForAxis(const SpaceSeparatedArray<LengthPercentage<Nonnegative>, 4>& values)
{
    bool isDefaultValue = false;

    SpaceSeparatedVector<LengthPercentage<Nonnegative>, 4>::Vector result;
    result.append(values.value[0]);

    if (values.value[3] != values.value[1]) {
        result.append(values.value[1]);
        result.append(values.value[2]);
        result.append(values.value[3]);
    } else if (values.value[2] != values.value[0]) {
        result.append(values.value[1]);
        result.append(values.value[2]);
    } else if (values.value[1] != values.value[0]) {
        result.append(values.value[1]);
    } else {
        isDefaultValue = result[0] == 0_css_px;
    }

    return { { WTFMove(result) }, isDefaultValue };
}

void Serialize<BorderRadius>::operator()(StringBuilder& builder, const BorderRadius& borderRadius)
{
    // <'border-radius'> = <length-percentage [0,âˆž]>{1,4} [ / <length-percentage [0,âˆž]>{1,4} ]?

    auto [horizontal, horizontalIsDefault] = gatherSerializableRadiiForAxis(borderRadius.horizontal);
    auto [vertical, verticalIsDefault] = gatherSerializableRadiiForAxis(borderRadius.vertical);

    if (!horizontalIsDefault || !verticalIsDefault) {
        serializationForCSS(builder, horizontal);

        if (horizontal != vertical) {
            builder.append(" / "_s);
            serializationForCSS(builder, vertical);
        }
    }
}

} // namespace CSS
} // namespace WebCore
