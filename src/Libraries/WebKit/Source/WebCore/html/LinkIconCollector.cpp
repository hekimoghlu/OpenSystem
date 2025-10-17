/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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
#include "LinkIconCollector.h"

#include "Document.h"
#include "ElementChildIteratorInlines.h"
#include "HTMLHeadElement.h"
#include "HTMLLinkElement.h"
#include "LinkIconType.h"
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

const unsigned defaultTouchIconWidth = 60;

static unsigned iconSize(const LinkIcon& icon)
{
    if (icon.size)
        return *icon.size;

    if (icon.type == LinkIconType::TouchIcon || icon.type == LinkIconType::TouchPrecomposedIcon)
        return defaultTouchIconWidth;

    return 0;
}

static int compareIcons(const LinkIcon& a, const LinkIcon& b)
{
    // Apple Touch icons always come first.
    if (a.type == LinkIconType::Favicon && b.type != LinkIconType::Favicon)
        return 1;
    if (b.type == LinkIconType::Favicon && a.type != LinkIconType::Favicon)
        return -1;

    unsigned aSize = iconSize(a);
    unsigned bSize = iconSize(b);

    if (bSize > aSize)
        return 1;
    if (bSize < aSize)
        return -1;

    // A Precomposed icon should come first if both icons have the same size.
    if (a.type != LinkIconType::TouchPrecomposedIcon && b.type == LinkIconType::TouchPrecomposedIcon)
        return 1;
    if (b.type != LinkIconType::TouchPrecomposedIcon && a.type == LinkIconType::TouchPrecomposedIcon)
        return -1;

    return 0;
}

auto LinkIconCollector::iconsOfTypes(OptionSet<LinkIconType> iconTypes) -> Vector<LinkIcon>
{
    RefPtr head = m_document.head();
    if (!head)
        return { };

    Vector<LinkIcon> icons;

    for (auto& linkElement : childrenOfType<HTMLLinkElement>(*head)) {
        if (!linkElement.iconType())
            continue;

        auto iconType = *linkElement.iconType();
        if (!iconTypes.contains(iconType))
            continue;

        auto url = linkElement.href();
        if (!url.protocolIsInHTTPFamily())
            continue;

        // This icon size parsing is a little wonky - it only parses the first
        // part of the size, "60x70" becomes "60". This is for compatibility reasons
        // and is probably good enough for now.
        std::optional<unsigned> iconSize;
        if (linkElement.sizes().length())
            iconSize = parseIntegerAllowingTrailingJunk<unsigned>(linkElement.sizes().item(0));

        Vector<std::pair<String, String>> attributes;
        if (linkElement.hasAttributes()) {
            auto linkAttributes = linkElement.attributes();
            attributes = WTF::map(linkAttributes, [](auto& attribute) -> std::pair<String, String> {
                return { attribute.localName(), attribute.value() };
            });
        }

        icons.append({ url, iconType, linkElement.type(), iconSize, WTFMove(attributes) });
    }

    std::sort(icons.begin(), icons.end(), [](auto& a, auto& b) {
        return compareIcons(a, b) < 0;
    });

    return icons;
}

}
