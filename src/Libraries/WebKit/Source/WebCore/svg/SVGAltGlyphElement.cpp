/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
#include "SVGAltGlyphElement.h"

#include "RenderInline.h"
#include "RenderSVGTSpan.h"
#include "SVGAltGlyphDefElement.h"
#include "SVGElementInlines.h"
#include "SVGElementTypeHelpers.h"
#include "SVGGlyphElement.h"
#include "SVGNames.h"
#include "XLinkNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGAltGlyphElement);

inline SVGAltGlyphElement::SVGAltGlyphElement(const QualifiedName& tagName, Document& document)
    : SVGTextPositioningElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , SVGURIReference(this)
{
    ASSERT(hasTagName(SVGNames::altGlyphTag));
}

Ref<SVGAltGlyphElement> SVGAltGlyphElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGAltGlyphElement(tagName, document));
}

ExceptionOr<void> SVGAltGlyphElement::setGlyphRef(const AtomString&)
{
    return Exception { ExceptionCode::NoModificationAllowedError };
}

const AtomString& SVGAltGlyphElement::glyphRef() const
{
    return attributeWithoutSynchronization(SVGNames::glyphRefAttr);
}

ExceptionOr<void> SVGAltGlyphElement::setFormat(const AtomString&)
{
    return Exception { ExceptionCode::NoModificationAllowedError };
}

const AtomString& SVGAltGlyphElement::format() const
{
    return attributeWithoutSynchronization(SVGNames::formatAttr);
}

bool SVGAltGlyphElement::childShouldCreateRenderer(const Node& child) const
{
    return child.isTextNode();
}

RenderPtr<RenderElement> SVGAltGlyphElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    return createRenderer<RenderSVGTSpan>(*this, WTFMove(style));
}

bool SVGAltGlyphElement::hasValidGlyphElements(Vector<String>& glyphNames) const
{
    // No need to support altGlyph referencing another node inside a shadow tree.
    auto target = targetElementFromIRIString(getAttribute(SVGNames::hrefAttr, XLinkNames::hrefAttr), document());

    if (is<SVGGlyphElement>(target.element)) {
        glyphNames.append(target.identifier);
        return true;
    }
    
    RefPtr altGlyphDefElement = downcast<SVGAltGlyphDefElement>(target.element.get());
    return altGlyphDefElement && altGlyphDefElement->hasValidGlyphElements(glyphNames);
}

}
