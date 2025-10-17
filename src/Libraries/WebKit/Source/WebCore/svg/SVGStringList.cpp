/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
#include "SVGStringList.h"

#include <wtf/text/StringBuilder.h>
#include <wtf/text/StringParsingBuffer.h>

namespace WebCore {

bool SVGStringList::parse(StringView data, UChar delimiter)
{
    clearItems();

    auto isSVGSpaceOrDelimiter = [delimiter](auto c) {
        return isASCIIWhitespace(c) || c == delimiter;
    };

    return readCharactersForParsing(data, [&](auto buffer) {
        skipOptionalSVGSpaces(buffer);

        while (buffer.hasCharactersRemaining()) {
            auto start = buffer.position();
            
            // FIXME: It would be a nice improvement to add a variant of skipUntil which worked
            // with lambda predicates.
            while (buffer.hasCharactersRemaining() && !isSVGSpaceOrDelimiter(*buffer))
                ++buffer;

            if (buffer.position() == start)
                break;

            m_items.append(String({ start, buffer.position() }));
            skipOptionalSVGSpacesOrDelimiter(buffer, delimiter);
        }

        // FIXME: Should this clearItems() on failure like SVGTransformList does?

        return buffer.atEnd();
    });
}

String SVGStringList::valueAsString() const
{
    StringBuilder builder;

    for (const auto& string : m_items) {
        if (builder.length())
            builder.append(' ');

        builder.append(string);
    }

    return builder.toString();
}

}
