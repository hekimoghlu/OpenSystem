/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
#include "SVGAnimationAdditiveValueFunctionImpl.h"

#include "RenderElement.h"
#include "SVGElement.h"
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

Color SVGAnimationColorFunction::colorFromString(SVGElement& targetElement, const String& string)
{
    static MainThreadNeverDestroyed<const AtomString> currentColor("currentColor"_s);

    if (string != currentColor.get())
        return SVGPropertyTraits<Color>::fromString(string);

    if (auto* renderer = targetElement.renderer())
        return renderer->style().visitedDependentColor(CSSPropertyColor);

    return { };
}

std::optional<float> SVGAnimationColorFunction::calculateDistance(SVGElement&, const String& from, const String& to) const
{
    auto simpleFrom = CSSParser::parseColorWithoutContext(from.trim(deprecatedIsSpaceOrNewline)).toColorTypeLossy<SRGBA<uint8_t>>().resolved();
    auto simpleTo = CSSParser::parseColorWithoutContext(to.trim(deprecatedIsSpaceOrNewline)).toColorTypeLossy<SRGBA<uint8_t>>().resolved();

    float red = simpleFrom.red - simpleTo.red;
    float green = simpleFrom.green - simpleTo.green;
    float blue = simpleFrom.blue - simpleTo.blue;

    return std::hypot(red, green, blue);
}

std::optional<float> SVGAnimationIntegerFunction::calculateDistance(SVGElement&, const String& from, const String& to) const
{
    return std::abs(parseInteger<int>(to).value_or(0) - parseInteger<int>(from).value_or(0));
}

} // namespace WebCore
