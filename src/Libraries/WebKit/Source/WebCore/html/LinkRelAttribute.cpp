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
#include "LinkRelAttribute.h"

#include "Document.h"
#include "LinkIconType.h"
#include "Settings.h"
#include <wtf/SortedArrayMap.h>
#include <wtf/text/StringView.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

// https://html.spec.whatwg.org/#linkTypes

struct LinkTypeDetails {
    bool (*isEnabled)(const Document&);
    void (*updateRel)(LinkRelAttribute&);
};

static constexpr std::pair<ComparableLettersLiteral, LinkTypeDetails> linkTypesArray[] = {
    { "alternate"_s, { [](auto) { return true; }, [](auto relAttribute) { relAttribute.isAlternate = true; } } },
    { "apple-touch-icon"_s, { [](auto) { return true; }, [](auto relAttribute) { relAttribute.iconType = LinkIconType::TouchIcon; } } },
    { "apple-touch-icon-precomposed"_s, { [](auto) { return true; }, [](auto relAttribute) { relAttribute.iconType = LinkIconType::TouchPrecomposedIcon; } } },
    { "dns-prefetch"_s, { [](auto) { return true; }, [](auto relAttribute) { relAttribute.isDNSPrefetch = true; } } },
    { "expect"_s, { [](auto) { return true; }, [](auto relAttribute) { relAttribute.isInternalResourceLink = true; } } },
    { "icon"_s, { [](auto) { return true; }, [](auto relAttribute) { relAttribute.iconType = LinkIconType::Favicon; } } },
#if ENABLE(APPLICATION_MANIFEST)
    { "manifest"_s, { [](auto) { return true; }, [](auto relAttribute) { relAttribute.isApplicationManifest = true; } } },
#else
    { "manifest"_s, { [](auto) { return false; }, [](auto) { } } },
#endif
    { "modulepreload"_s, { [](auto) { return true; }, [](auto relAttribute) { relAttribute.isLinkModulePreload = true; } } },
    { "preconnect"_s, { [](auto document) { return document.settings().linkPreconnectEnabled(); }, [](auto relAttribute) { relAttribute.isLinkPreconnect = true; } } },
    { "prefetch"_s, { [](auto document) { return document.settings().linkPrefetchEnabled(); }, [](auto relAttribute) { relAttribute.isLinkPrefetch = true; } } },
    { "preload"_s, { [](auto document) { return document.settings().linkPreloadEnabled(); }, [](auto relAttribute) { relAttribute.isLinkPreload = true; } } },
    { "stylesheet"_s, { [](auto) { return true; }, [](auto relAttribute) { relAttribute.isStyleSheet = true; } } },
};

static constexpr SortedArrayMap linkTypes { linkTypesArray };

LinkRelAttribute::LinkRelAttribute(Document& document, StringView rel)
{
    if (auto linkType = linkTypes.tryGet(rel)) {
        if (linkType->isEnabled(document))
            linkType->updateRel(*this);
        return;
    }
    if (equalLettersIgnoringASCIICase(rel, "shortcut icon"_s))
        iconType = LinkIconType::Favicon;
    else if (equalLettersIgnoringASCIICase(rel, "alternate stylesheet"_s) || equalLettersIgnoringASCIICase(rel, "stylesheet alternate"_s)) {
        isStyleSheet = true;
        isAlternate = true;
    } else {
        // Tokenize the rel attribute and set bits based on specific keywords that we find.
        unsigned length = rel.length();
        unsigned start = 0;
        while (start < length) {
            if (isASCIIWhitespace(rel[start])) {
                start++;
                continue;
            }
            unsigned end = start + 1;
            while (end < length && !isASCIIWhitespace(rel[end]))
                end++;
            if (auto linkType = linkTypes.tryGet(rel.substring(start, end - start))) {
                if (linkType->isEnabled(document))
                    linkType->updateRel(*this);
            }
            start = end;
        }
    }
}

bool LinkRelAttribute::isSupported(Document& document, StringView attribute)
{
    if (auto linkType = linkTypes.tryGet(attribute))
        return linkType->isEnabled(document);
    return false;
}

}
