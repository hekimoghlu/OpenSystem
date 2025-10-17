/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#include "MathMLAnnotationElement.h"

#if ENABLE(MATHML)

#include "ElementInlines.h"
#include "HTMLHtmlElement.h"
#include "MathMLMathElement.h"
#include "MathMLNames.h"
#include "MathMLSelectElement.h"
#include "RenderMathMLBlock.h"
#include "SVGElementTypeHelpers.h"
#include "SVGSVGElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLAnnotationElement);

using namespace MathMLNames;

MathMLAnnotationElement::MathMLAnnotationElement(const QualifiedName& tagName, Document& document)
    : MathMLPresentationElement(tagName, document)
{
    ASSERT(hasTagName(annotationTag) || hasTagName(annotation_xmlTag));
}

Ref<MathMLAnnotationElement> MathMLAnnotationElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLAnnotationElement(tagName, document));
}

RenderPtr<RenderElement> MathMLAnnotationElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition& insertionPosition)
{
    if (hasTagName(MathMLNames::annotationTag))
        return MathMLElement::createElementRenderer(WTFMove(style), insertionPosition);

    ASSERT(hasTagName(annotation_xmlTag));
    return createRenderer<RenderMathMLBlock>(RenderObject::Type::MathMLBlock, *this, WTFMove(style));
}

bool MathMLAnnotationElement::childShouldCreateRenderer(const Node& child) const
{
    // For <annotation>, only text children are allowed.
    if (hasTagName(MathMLNames::annotationTag))
        return child.isTextNode();

    ASSERT(hasTagName(annotation_xmlTag));
    return StyledElement::childShouldCreateRenderer(child);
}

void MathMLAnnotationElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason reason)
{
    if (name == MathMLNames::srcAttr || name == MathMLNames::encodingAttr) {
        RefPtr parent = parentElement();
        if (is<MathMLElement>(parent) && parent->hasTagName(semanticsTag))
            downcast<MathMLElement>(*parent).updateSelectedChild();
    }
    MathMLPresentationElement::attributeChanged(name, oldValue, newValue, reason);
}

}

#endif // ENABLE(MATHML)
