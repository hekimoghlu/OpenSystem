/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#include "SVGPolyElement.h"

#include "DocumentInlines.h"
#include "LegacyRenderSVGPath.h"
#include "LegacyRenderSVGResource.h"
#include "RenderSVGPath.h"
#include "SVGDocumentExtensions.h"
#include "SVGParserUtilities.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGPolyElement);

SVGPolyElement::SVGPolyElement(const QualifiedName& tagName, Document& document)
    : SVGGeometryElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::pointsAttr, &SVGPolyElement::m_points>();
    });
}

void SVGPolyElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == SVGNames::pointsAttr) {
        if (!m_points->baseVal()->parse(newValue))
            protectedDocument()->checkedSVGExtensions()->reportError(makeString("Problem parsing points=\""_s, newValue, "\""_s));
    }

    SVGGeometryElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGPolyElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        ASSERT(attrName == SVGNames::pointsAttr);
        InstanceInvalidationGuard guard(*this);

        if (CheckedPtr path = dynamicDowncast<RenderSVGPath>(renderer()))
            path->setNeedsShapeUpdate();

        if (CheckedPtr path = dynamicDowncast<LegacyRenderSVGPath>(renderer()))
            path->setNeedsShapeUpdate();

        updateSVGRendererForElementChange();
        invalidateResourceImageBuffersIfNeeded();
        return;
    }

    SVGGeometryElement::svgAttributeChanged(attrName);
}

size_t SVGPolyElement::approximateMemoryCost() const
{
    size_t pointsCost = m_points->baseVal()->items().size() * sizeof(FloatPoint);

    if (document().settings().layerBasedSVGEngineEnabled()) {
        // We need to account for the memory which is allocated by the RenderSVGPath::m_path.
        return sizeof(*this) + (renderer() ? pointsCost * 2 + sizeof(RenderSVGPath) : pointsCost);
    }

    // We need to account for the memory which is allocated by the LegacyRenderSVGPath::m_path.
    return sizeof(*this) + (renderer() ? pointsCost * 2 + sizeof(LegacyRenderSVGPath) : pointsCost);
}

}
