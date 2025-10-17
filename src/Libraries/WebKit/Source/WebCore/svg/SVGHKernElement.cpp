/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#include "SVGHKernElement.h"

#include "SVGElementInlines.h"
#include "SVGFontElement.h"
#include "SVGFontFaceElement.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGHKernElement);

inline SVGHKernElement::SVGHKernElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::hkernTag));
}

Ref<SVGHKernElement> SVGHKernElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGHKernElement(tagName, document));
}

std::optional<SVGKerningPair> SVGHKernElement::buildHorizontalKerningPair() const
{
    // FIXME: Can this be shared with SVGVKernElement::buildVerticalKerningPair?

    auto& u1 = attributeWithoutSynchronization(SVGNames::u1Attr);
    auto& g1 = attributeWithoutSynchronization(SVGNames::g1Attr);
    if (u1.isEmpty() && g1.isEmpty())
        return std::nullopt;

    auto& u2 = attributeWithoutSynchronization(SVGNames::u2Attr);
    auto& g2 = attributeWithoutSynchronization(SVGNames::g2Attr);
    if (u2.isEmpty() && g2.isEmpty())
        return std::nullopt;

    auto glyphName1 = parseGlyphName(g1);
    if (!glyphName1)
        return std::nullopt;
    auto glyphName2 = parseGlyphName(g2);
    if (!glyphName2)
        return std::nullopt;
    auto unicodeString1 = parseKerningUnicodeString(u1);
    if (!unicodeString1)
        return std::nullopt;
    auto unicodeString2 = parseKerningUnicodeString(u2);
    if (!unicodeString2)
        return std::nullopt;

    bool ok = false;
    auto kerning = attributeWithoutSynchronization(SVGNames::kAttr).string().toFloat(&ok);
    if (!ok)
        return std::nullopt;

    return SVGKerningPair {
        WTFMove(unicodeString1->first),
        WTFMove(unicodeString1->second),
        WTFMove(*glyphName1),
        WTFMove(unicodeString2->first),
        WTFMove(unicodeString2->second),
        WTFMove(*glyphName2),
        kerning
    };
}

}
