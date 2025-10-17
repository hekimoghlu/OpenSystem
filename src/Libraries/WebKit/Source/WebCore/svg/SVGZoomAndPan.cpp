/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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
#include "SVGZoomAndPan.h"

#include <wtf/text/StringParsingBuffer.h>

namespace WebCore {

template<typename CharacterType> static constexpr std::array<CharacterType, 7> disable { 'd', 'i', 's', 'a', 'b', 'l', 'e' };
template<typename CharacterType> static constexpr std::array<CharacterType, 7> magnify { 'm', 'a', 'g', 'n', 'i', 'f', 'y' };

template<typename CharacterType> static std::optional<SVGZoomAndPanType> parseZoomAndPanGeneric(StringParsingBuffer<CharacterType>& buffer)
{
    if (skipCharactersExactly(buffer, std::span { disable<CharacterType> }))
        return SVGZoomAndPanDisable;

    if (skipCharactersExactly(buffer, std::span { magnify<CharacterType> }))
        return SVGZoomAndPanMagnify;

    return std::nullopt;
}

std::optional<SVGZoomAndPanType> SVGZoomAndPan::parseZoomAndPan(StringParsingBuffer<LChar>& buffer)
{
    return parseZoomAndPanGeneric(buffer);
}

std::optional<SVGZoomAndPanType> SVGZoomAndPan::parseZoomAndPan(StringParsingBuffer<UChar>& buffer)
{
    return parseZoomAndPanGeneric(buffer);
}

void SVGZoomAndPan::parseAttribute(const QualifiedName& attributeName, const AtomString& value)
{
    if (attributeName != SVGNames::zoomAndPanAttr)
        return;
    m_zoomAndPan = SVGPropertyTraits<SVGZoomAndPanType>::fromString(value);
}

}
