/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#include "SVGGeometryElement.h"

#include "DOMPoint.h"
#include "DocumentInlines.h"
#include "LegacyRenderSVGResource.h"
#include "LegacyRenderSVGShape.h"
#include "RenderSVGShape.h"
#include "SVGDocumentExtensions.h"
#include "SVGPathUtilities.h"
#include "SVGPoint.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGGeometryElement);

SVGGeometryElement::SVGGeometryElement(const QualifiedName& tagName, Document& document, UniqueRef<SVGPropertyRegistry>&& propertyRegistry)
    : SVGGraphicsElement(tagName, document, WTFMove(propertyRegistry))
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::pathLengthAttr, &SVGGeometryElement::m_pathLength>();
    });
}

float SVGGeometryElement::getTotalLength() const
{
    protectedDocument()->updateLayoutIgnorePendingStylesheets({ LayoutOptions::ContentVisibilityForceLayout }, this);

    auto* renderer = this->renderer();
    if (!renderer)
        return 0;

    if (CheckedPtr renderSVGShape = dynamicDowncast<LegacyRenderSVGShape>(renderer))
        return renderSVGShape->getTotalLength();

    if (CheckedPtr renderSVGShape = dynamicDowncast<RenderSVGShape>(renderer))
        return renderSVGShape->getTotalLength();

    ASSERT_NOT_REACHED();
    return 0;
}

ExceptionOr<Ref<SVGPoint>> SVGGeometryElement::getPointAtLength(float distance) const
{
    protectedDocument()->updateLayoutIgnorePendingStylesheets({ LayoutOptions::ContentVisibilityForceLayout }, this);

    auto* renderer = this->renderer();
    // Spec: If current element is a non-rendered element, throw an InvalidStateError.
    if (!renderer)
        return Exception { ExceptionCode::InvalidStateError };

    // Spec: Clamp distance to [0, length].
    distance = clampTo<float>(distance, 0, getTotalLength());

    // Spec: Return a newly created, detached SVGPoint object.
    if (CheckedPtr renderSVGShape = dynamicDowncast<LegacyRenderSVGShape>(renderer))
        return SVGPoint::create(renderSVGShape->getPointAtLength(distance));

    if (CheckedPtr renderSVGShape = dynamicDowncast<RenderSVGShape>(renderer))
        return SVGPoint::create(renderSVGShape->getPointAtLength(distance));

    ASSERT_NOT_REACHED();
    return Exception { ExceptionCode::InvalidStateError };
}

bool SVGGeometryElement::isPointInFill(DOMPointInit&& pointInit)
{
    protectedDocument()->updateLayoutIgnorePendingStylesheets({ LayoutOptions::ContentVisibilityForceLayout }, this);

    auto* renderer = this->renderer();
    if (!renderer)
        return false;

    FloatPoint point {static_cast<float>(pointInit.x), static_cast<float>(pointInit.y)};
    if (CheckedPtr renderSVGShape = dynamicDowncast<LegacyRenderSVGShape>(renderer))
        return renderSVGShape->isPointInFill(point);

    if (CheckedPtr renderSVGShape = dynamicDowncast<RenderSVGShape>(renderer))
        return renderSVGShape->isPointInFill(point);

    ASSERT_NOT_REACHED();
    return false;
}

bool SVGGeometryElement::isPointInStroke(DOMPointInit&& pointInit)
{
    protectedDocument()->updateLayoutIgnorePendingStylesheets({ LayoutOptions::ContentVisibilityForceLayout }, this);

    auto* renderer = this->renderer();
    if (!renderer)
        return false;

    FloatPoint point {static_cast<float>(pointInit.x), static_cast<float>(pointInit.y)};
    if (CheckedPtr renderSVGShape = dynamicDowncast<LegacyRenderSVGShape>(renderer))
        return renderSVGShape->isPointInStroke(point);

    if (CheckedPtr renderSVGShape = dynamicDowncast<RenderSVGShape>(renderer))
        return renderSVGShape->isPointInStroke(point);

    ASSERT_NOT_REACHED();
    return false;
}

void SVGGeometryElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == SVGNames::pathLengthAttr) {
        Ref pathLength = m_pathLength;
        pathLength->setBaseValInternal(newValue.toFloat());
        if (pathLength->baseVal() < 0)
            protectedDocument()->checkedSVGExtensions()->reportError("A negative value for path attribute <pathLength> is not allowed"_s);
    }

    SVGGraphicsElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGGeometryElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        ASSERT(attrName == SVGNames::pathLengthAttr);
        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        return;
    }

    SVGGraphicsElement::svgAttributeChanged(attrName);
}

}
