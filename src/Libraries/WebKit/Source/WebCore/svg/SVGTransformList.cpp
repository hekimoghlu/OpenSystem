/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
#include "SVGTransformList.h"

#include "SVGParserUtilities.h"
#include <wtf/text/StringBuilder.h>
#include <wtf/text/StringParsingBuffer.h>

namespace WebCore {

ExceptionOr<RefPtr<SVGTransform>> SVGTransformList::consolidate()
{
    auto result = canAlterList();
    if (result.hasException())
        return result.releaseException();
    ASSERT(result.releaseReturnValue());

    // Spec: If the list was empty, then a value of null is returned.
    if (m_items.isEmpty())
        return nullptr;

    if (m_items.size() == 1)
        return RefPtr { at(0).ptr() };

    auto newItem = SVGTransform::create(concatenate());
    clearItems();

    auto item = append(WTFMove(newItem));
    commitChange();
    return RefPtr { item.ptr() };
}

AffineTransform SVGTransformList::concatenate() const
{
    AffineTransform result;
    for (auto& transform : m_items)
        result *= transform->matrix().value();
    return result;
}

template<typename CharacterType> bool SVGTransformList::parseGeneric(StringParsingBuffer<CharacterType>& buffer)
{
    bool delimParsed = false;
    skipOptionalSVGSpaces(buffer);

    while (buffer.hasCharactersRemaining()) {
        delimParsed = false;
        
        auto parsedTransformType = SVGTransformable::parseTransformType(buffer);
        if (!parsedTransformType)
            return false;

        auto parsedTransformValue = SVGTransformable::parseTransformValue(*parsedTransformType, buffer);
        if (!parsedTransformValue)
            return false;

        append(SVGTransform::create(WTFMove(*parsedTransformValue)));

        skipOptionalSVGSpaces(buffer);

        if (skipExactly(buffer, ','))
            delimParsed = true;

        skipOptionalSVGSpaces(buffer);
    }
    return !delimParsed;
}

void SVGTransformList::parse(StringView value)
{
    clearItems();

    bool parsingSucceeded = readCharactersForParsing(value, [&](auto buffer) {
        return parseGeneric(buffer);
    });
    
    if (!parsingSucceeded)
        clearItems();
}

bool SVGTransformList::parse(StringParsingBuffer<LChar>& buffer)
{
    return parseGeneric(buffer);
}

bool SVGTransformList::parse(StringParsingBuffer<UChar>& buffer)
{
    return parseGeneric(buffer);
}

String SVGTransformList::valueAsString() const
{
    StringBuilder builder;
    for (const auto& transform : m_items) {
        if (builder.length())
            builder.append(' ');

        builder.append(transform->value().valueAsString());
    }
    return builder.toString();
}

}

