/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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
#include "SVGTSpanElement.h"

#include "RenderInline.h"
#include "RenderSVGTSpan.h"
#include "SVGElementInlines.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGTSpanElement);

inline SVGTSpanElement::SVGTSpanElement(const QualifiedName& tagName, Document& document)
    : SVGTextPositioningElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::tspanTag));
}

Ref<SVGTSpanElement> SVGTSpanElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGTSpanElement(tagName, document));
}

RenderPtr<RenderElement> SVGTSpanElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    return createRenderer<RenderSVGTSpan>(*this, WTFMove(style));
}

bool SVGTSpanElement::childShouldCreateRenderer(const Node& child) const
{
    if (child.isTextNode()
        || child.hasTagName(SVGNames::aTag)
        || child.hasTagName(SVGNames::altGlyphTag)
        || child.hasTagName(SVGNames::trefTag)
        || child.hasTagName(SVGNames::tspanTag))
        return true;

    return false;
}

bool SVGTSpanElement::rendererIsNeeded(const RenderStyle& style)
{
    if (parentNode()
        && (parentNode()->hasTagName(SVGNames::aTag)
            || parentNode()->hasTagName(SVGNames::altGlyphTag)
            || parentNode()->hasTagName(SVGNames::textTag)
            || parentNode()->hasTagName(SVGNames::textPathTag)
            || parentNode()->hasTagName(SVGNames::tspanTag)))
        return StyledElement::rendererIsNeeded(style);

    return false;
}

}
