/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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
#include "SVGTransformable.h"

#include "AffineTransform.h"
#include "FloatConversion.h"
#include "SVGElement.h"
#include "SVGNames.h"
#include "SVGParserUtilities.h"
#include <wtf/text/StringParsingBuffer.h>
#include <wtf/text/StringView.h>

namespace WebCore {

SVGTransformable::~SVGTransformable() = default;

template<typename CharacterType> static int parseTransformParamList(StringParsingBuffer<CharacterType>& buffer, std::span<float, 6> values, int required, int optional)
{
    int optionalParams = 0, requiredParams = 0;
    
    if (!skipOptionalSVGSpaces(buffer) || *buffer != '(')
        return -1;
    
    ++buffer;
   
    skipOptionalSVGSpaces(buffer);

    while (requiredParams < required) {
        if (buffer.atEnd())
            return -1;
        auto parsedNumber = parseNumber(buffer, SuffixSkippingPolicy::DontSkip);
        if (!parsedNumber)
            return -1;
        values[requiredParams] = *parsedNumber;
        requiredParams++;
        if (requiredParams < required)
            skipOptionalSVGSpacesOrDelimiter(buffer);
    }
    if (!skipOptionalSVGSpaces(buffer))
        return -1;
    
    bool delimParsed = skipOptionalSVGSpacesOrDelimiter(buffer);

    if (buffer.atEnd())
        return -1;
    
    if (*buffer == ')') {
        // skip optionals
        ++buffer;
        if (delimParsed)
            return -1;
    } else {
        while (optionalParams < optional) {
            if (buffer.atEnd())
                return -1;
            auto parsedNumber = parseNumber(buffer, SuffixSkippingPolicy::DontSkip);
            if (!parsedNumber)
                return -1;
            values[requiredParams + optionalParams] = *parsedNumber;
            optionalParams++;
            if (optionalParams < optional)
                skipOptionalSVGSpacesOrDelimiter(buffer);
        }
        
        if (!skipOptionalSVGSpaces(buffer))
            return -1;
        
        delimParsed = skipOptionalSVGSpacesOrDelimiter(buffer);
        
        if (buffer.atEnd() || *buffer != ')' || delimParsed)
            return -1;
        ++buffer;
    }

    return requiredParams + optionalParams;
}

// These should be kept in sync with enum SVGTransformType
static constexpr std::array requiredValuesForType { 0, 6, 1, 1, 1, 1, 1 };
static constexpr std::array optionalValuesForType { 0, 0, 1, 1, 2, 0, 0 };

template<typename CharacterType> static std::optional<SVGTransformValue> parseTransformValueGeneric(SVGTransformValue::SVGTransformType type, StringParsingBuffer<CharacterType>& buffer)
{
    if (type == SVGTransformValue::SVG_TRANSFORM_UNKNOWN)
        return std::nullopt;

    int valueCount = 0;
    std::array<float, 6> values { 0, 0, 0, 0, 0, 0 };
    if ((valueCount = parseTransformParamList(buffer, values, requiredValuesForType[type], optionalValuesForType[type])) < 0)
        return std::nullopt;

    switch (type) {
    case SVGTransformValue::SVG_TRANSFORM_UNKNOWN:
        ASSERT_NOT_REACHED();
        return std::nullopt;

    case SVGTransformValue::SVG_TRANSFORM_SKEWX: {
        SVGTransformValue transform;
        transform.setSkewX(values[0]);
        return transform;
    }
    case SVGTransformValue::SVG_TRANSFORM_SKEWY: {
        SVGTransformValue transform;
        transform.setSkewY(values[0]);
        return transform;
    }
    case SVGTransformValue::SVG_TRANSFORM_SCALE:
        if (valueCount == 1) // Spec: if only one param given, assume uniform scaling
            return SVGTransformValue::scaleTransformValue({ values[0], values[0] });

        return SVGTransformValue::scaleTransformValue({ values[0], values[1] });

    case SVGTransformValue::SVG_TRANSFORM_TRANSLATE:
        if (valueCount == 1) // Spec: if only one param given, assume 2nd param to be 0
            return SVGTransformValue::translateTransformValue({ values[0], 0 });

        return SVGTransformValue::translateTransformValue({ values[0], values[1] });

    case SVGTransformValue::SVG_TRANSFORM_ROTATE:
        if (valueCount == 1)
            return SVGTransformValue::rotateTransformValue(values[0], { });

        return SVGTransformValue::rotateTransformValue(values[0], { values[1], values[2] });

    case SVGTransformValue::SVG_TRANSFORM_MATRIX: {
        SVGTransformValue transform;
        transform.setMatrix(AffineTransform(values[0], values[1], values[2], values[3], values[4], values[5]));
        return transform;
    }
    }

    return std::nullopt;
}

std::optional<SVGTransformValue> SVGTransformable::parseTransformValue(SVGTransformValue::SVGTransformType type, StringParsingBuffer<LChar>& buffer)
{
    return parseTransformValueGeneric(type, buffer);
}

std::optional<SVGTransformValue> SVGTransformable::parseTransformValue(SVGTransformValue::SVGTransformType type, StringParsingBuffer<UChar>& buffer)
{
    return parseTransformValueGeneric(type, buffer);
}

template<typename CharacterType> static constexpr std::array<CharacterType, 5> skewXDesc  { 's', 'k', 'e', 'w', 'X' };
template<typename CharacterType> static constexpr std::array<CharacterType, 5> skewYDesc  { 's', 'k', 'e', 'w', 'Y' };
template<typename CharacterType> static constexpr std::array<CharacterType, 5> scaleDesc  { 's', 'c', 'a', 'l', 'e' };
template<typename CharacterType> static constexpr std::array<CharacterType, 9> translateDesc  { 't', 'r', 'a', 'n', 's', 'l', 'a', 't', 'e' };
template<typename CharacterType> static constexpr std::array<CharacterType, 6> rotateDesc  { 'r', 'o', 't', 'a', 't', 'e' };
template<typename CharacterType> static constexpr std::array<CharacterType, 6> matrixDesc  { 'm', 'a', 't', 'r', 'i', 'x' };

template<typename CharacterType> static std::optional<SVGTransformValue::SVGTransformType> parseTransformTypeGeneric(StringParsingBuffer<CharacterType>& buffer)
{
    if (buffer.atEnd())
        return std::nullopt;

    if (*buffer == 's') {
        if (skipCharactersExactly(buffer, std::span { skewXDesc<CharacterType> }))
            return SVGTransformValue::SVG_TRANSFORM_SKEWX;
        if (skipCharactersExactly(buffer, std::span { skewYDesc<CharacterType> }))
            return SVGTransformValue::SVG_TRANSFORM_SKEWY;
        if (skipCharactersExactly(buffer, std::span { scaleDesc<CharacterType> }))
            return SVGTransformValue::SVG_TRANSFORM_SCALE;
        return std::nullopt;
    }

    if (skipCharactersExactly(buffer, std::span { translateDesc<CharacterType> }))
        return SVGTransformValue::SVG_TRANSFORM_TRANSLATE;
    if (skipCharactersExactly(buffer, std::span { rotateDesc<CharacterType> }))
        return SVGTransformValue::SVG_TRANSFORM_ROTATE;
    if (skipCharactersExactly(buffer, std::span { matrixDesc<CharacterType> }))
        return SVGTransformValue::SVG_TRANSFORM_MATRIX;

    return std::nullopt;
}

std::optional<SVGTransformValue::SVGTransformType> SVGTransformable::parseTransformType(StringView string)
{
    return readCharactersForParsing(string, [](auto buffer) {
        return parseTransformType(buffer);
    });
}

std::optional<SVGTransformValue::SVGTransformType> SVGTransformable::parseTransformType(StringParsingBuffer<LChar>& buffer)
{
    return parseTransformTypeGeneric(buffer);
}

std::optional<SVGTransformValue::SVGTransformType> SVGTransformable::parseTransformType(StringParsingBuffer<UChar>& buffer)
{
    return parseTransformTypeGeneric(buffer);
}

}
