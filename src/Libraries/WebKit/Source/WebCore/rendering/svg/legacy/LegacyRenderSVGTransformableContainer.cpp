/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#include "LegacyRenderSVGTransformableContainer.h"

#include "RenderElementInlines.h"
#include "RenderStyleInlines.h"
#include "SVGElementTypeHelpers.h"
#include "SVGGElement.h"
#include "SVGGraphicsElement.h"
#include "SVGUseElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGTransformableContainer);

LegacyRenderSVGTransformableContainer::LegacyRenderSVGTransformableContainer(SVGGraphicsElement& element, RenderStyle&& style)
    : LegacyRenderSVGContainer(Type::LegacySVGTransformableContainer, element, WTFMove(style))
    , m_needsTransformUpdate(true)
    , m_didTransformToRootUpdate(false)
{
    ASSERT(isLegacyRenderSVGTransformableContainer());
}

LegacyRenderSVGTransformableContainer::~LegacyRenderSVGTransformableContainer() = default;

bool LegacyRenderSVGTransformableContainer::calculateLocalTransform()
{
    Ref element = graphicsElement();

    // If we're either the renderer for a <use> element, or for any <g> element inside the shadow
    // tree, that was created during the use/symbol/svg expansion in SVGUseElement. These containers
    // need to respect the translations induced by their corresponding use elements x/y attributes.
    RefPtr useElement = dynamicDowncast<SVGUseElement>(element.get());
    if (!useElement && element->isInShadowTree() && is<SVGGElement>(element)) {
        if (auto* correspondingElement = dynamicDowncast<SVGUseElement>(element->correspondingElement()))
            useElement = correspondingElement;
    }

    if (useElement) {
        SVGLengthContext lengthContext(element.ptr());
        FloatSize translation(useElement->x().value(lengthContext), useElement->y().value(lengthContext));
        if (translation != m_lastTranslation)
            m_needsTransformUpdate = true;
        m_lastTranslation = translation;
    }

    auto referenceBoxRect = transformReferenceBoxRect();
    if (referenceBoxRect != m_lastTransformReferenceBoxRect) {
        m_lastTransformReferenceBoxRect = referenceBoxRect;
        m_needsTransformUpdate = true;
    }

    m_didTransformToRootUpdate = m_needsTransformUpdate || SVGRenderSupport::transformToRootChanged(parent());
    if (!m_needsTransformUpdate)
        return false;

    m_localTransform = element->animatedLocalTransform();
    m_localTransform.translate(m_lastTranslation);
    m_needsTransformUpdate = false;
    return true;
}

SVGGraphicsElement& LegacyRenderSVGTransformableContainer::graphicsElement()
{
    return downcast<SVGGraphicsElement>(LegacyRenderSVGContainer::element());
}

}
