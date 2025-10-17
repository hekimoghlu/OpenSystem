/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#include "SVGLengthList.h"

#include "EventTarget.h"
#include <wtf/text/StringBuilder.h>
#include <wtf/text/StringParsingBuffer.h>

namespace WebCore {

bool SVGLengthList::parse(StringView value)
{
    clearItems();

    return readCharactersForParsing(value, [&](auto buffer) {
        skipOptionalSVGSpaces(buffer);

        while (buffer.hasCharactersRemaining()) {
            auto start = buffer.position();

            skipUntil<isSVGSpaceOrComma>(buffer);

            if (buffer.position() == start)
                break;

            auto value = SVGLengthValue::construct(m_lengthMode, std::span(start, buffer.position() - start));
            if (!value)
                break;

            append(SVGLength::create(WTFMove(*value)));
            skipOptionalSVGSpacesOrDelimiter(buffer);
        }

        // FIXME: Should this clearItems() on failure like SVGTransformList does?

        return buffer.atEnd();
    });
}

String SVGLengthList::valueAsString() const
{
    StringBuilder builder;

    for (const auto& length : m_items) {
        if (builder.length())
            builder.append(' ');

        builder.append(length->value().valueAsString());
    }

    return builder.toString();
}

}
