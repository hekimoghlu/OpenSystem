/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
#pragma once

#include "SVGGeometryElement.h"
#include "SVGNames.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGPolyElement : public SVGGeometryElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGPolyElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGPolyElement);
public:
    const SVGPointList& points() const { return m_points->currentValue(); }

    SVGPointList& points() { return m_points->baseVal(); }
    SVGPointList& animatedPoints() { return *m_points->animVal(); }

    size_t approximateMemoryCost() const override;

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGPolyElement, SVGGeometryElement>;

protected:
    SVGPolyElement(const QualifiedName&, Document&);

private:
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool isValid() const override { return SVGTests::isValid(); }
    bool supportsMarkers() const override { return true; }

    Ref<SVGAnimatedPointList> m_points { SVGAnimatedPointList::create(this) };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SVGPolyElement)
    static bool isType(const WebCore::SVGElement& element) { return element.hasTagName(WebCore::SVGNames::polygonTag) || element.hasTagName(WebCore::SVGNames::polylineTag); }
    static bool isType(const WebCore::Node& node)
    {
        auto* svgElement = dynamicDowncast<WebCore::SVGElement>(node);
        return svgElement && isType(*svgElement);
    }
SPECIALIZE_TYPE_TRAITS_END()
