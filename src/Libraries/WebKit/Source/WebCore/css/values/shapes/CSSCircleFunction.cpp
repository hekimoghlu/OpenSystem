/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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
#include "CSSCircleFunction.h"

#include "CSSPrimitiveNumericTypes+Serialization.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace CSS {

static bool hasDefaultValueForCircleRadius(Circle::RadialSize radius)
{
    return WTF::switchOn(radius,
        [](Circle::Extent extent) {
            // FIXME: The spec says that `farthest-corner` should be the default, but this does not match the tests.
            return std::holds_alternative<Keyword::ClosestSide>(extent);
        },
        [](const auto&) {
            return false;
        }
    );
}

void Serialize<Circle>::operator()(StringBuilder& builder, const Circle& value)
{
    // <circle()> = circle( <radial-size>? [ at <position> ]? )

    auto lengthBefore = builder.length();

    if (!hasDefaultValueForCircleRadius(value.radius))
        serializationForCSS(builder, value.radius);

    if (value.position) {
        // FIXME: To match other serialization of Percentage, this should not serialize if equal to the default value of 50% 50%, but this does not match the tests.
        bool wroteSomething = builder.length() != lengthBefore;
        builder.append(wroteSomething ? " at "_s : "at "_s);
        serializationForCSS(builder, *value.position);
    }
}

} // namespace CSS
} // namespace WebCore
