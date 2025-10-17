/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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
#include "SVGGElement.h"

#include "LegacyRenderSVGHiddenContainer.h"
#include "LegacyRenderSVGResource.h"
#include "LegacyRenderSVGTransformableContainer.h"
#include "RenderSVGHiddenContainer.h"
#include "RenderSVGTransformableContainer.h"
#include "SVGNames.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGGElement);

SVGGElement::SVGGElement(const QualifiedName& tagName, Document& document)
    : SVGGraphicsElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::gTag));
}

Ref<SVGGElement> SVGGElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGGElement(tagName, document));
}

Ref<SVGGElement> SVGGElement::create(Document& document)
{
    return create(SVGNames::gTag, document);
}

RenderPtr<RenderElement> SVGGElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    // FIXME: [LBSE] Support hidden containers
    if (document().settings().layerBasedSVGEngineEnabled()) {
        if (style.display() == DisplayType::None)
            return createRenderer<RenderSVGHiddenContainer>(RenderObject::Type::SVGHiddenContainer, *this, WTFMove(style));
        return createRenderer<RenderSVGTransformableContainer>(*this, WTFMove(style));
    }

    // SVG 1.1 testsuite explicitly uses constructs like <g display="none"><linearGradient>
    // We still have to create renderers for the <g> & <linearGradient> element, though the
    // subtree may be hidden - we only want the resource renderers to exist so they can be
    // referenced from somewhere else.
    if (style.display() == DisplayType::None)
        return createRenderer<LegacyRenderSVGHiddenContainer>(RenderObject::Type::LegacySVGHiddenContainer, *this, WTFMove(style));
    return createRenderer<LegacyRenderSVGTransformableContainer>(*this, WTFMove(style));
}

bool SVGGElement::rendererIsNeeded(const RenderStyle&)
{
    // Unlike SVGElement::rendererIsNeeded(), we still create renderers, even if
    // display is set to 'none' - which is special to SVG <g> container elements.
    return parentOrShadowHostElement() && parentOrShadowHostElement()->isSVGElement();
}

}
