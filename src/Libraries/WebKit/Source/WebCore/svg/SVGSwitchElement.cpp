/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
#include "SVGSwitchElement.h"

#include "ElementChildIteratorInlines.h"
#include "LegacyRenderSVGTransformableContainer.h"
#include "RenderSVGTransformableContainer.h"
#include "SVGElementTypeHelpers.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGSwitchElement);

inline SVGSwitchElement::SVGSwitchElement(const QualifiedName& tagName, Document& document)
    : SVGGraphicsElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::switchTag));
}

Ref<SVGSwitchElement> SVGSwitchElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGSwitchElement(tagName, document));
}

bool SVGSwitchElement::childShouldCreateRenderer(const Node& child) const
{
    // We create a renderer for the first valid SVG element child.
    // FIXME: The renderer must be updated after dynamic change of the requiredFeatures, requiredExtensions and systemLanguage attributes (https://bugs.webkit.org/show_bug.cgi?id=74749).
    for (auto& element : childrenOfType<SVGElement>(*this)) {
        if (!element.isValid())
            continue;
        return &element == &child; // Only allow this child if it's the first valid child
    }

    return false;
}

RenderPtr<RenderElement> SVGSwitchElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    if (document().settings().layerBasedSVGEngineEnabled())
        return createRenderer<RenderSVGTransformableContainer>(*this, WTFMove(style));
    return createRenderer<LegacyRenderSVGTransformableContainer>(*this, WTFMove(style));
}

}
