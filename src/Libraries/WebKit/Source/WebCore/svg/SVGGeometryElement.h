/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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

#include "Path.h"
#include "SVGGraphicsElement.h"
#include "SVGNames.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct DOMPointInit;
class SVGPoint;

class SVGGeometryElement : public SVGGraphicsElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGGeometryElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGGeometryElement);
public:
    virtual float getTotalLength() const;
    virtual ExceptionOr<Ref<SVGPoint>> getPointAtLength(float distance) const;

    bool isPointInFill(DOMPointInit&&);
    bool isPointInStroke(DOMPointInit&&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGGeometryElement, SVGGraphicsElement>;

    float pathLength() const { return m_pathLength->currentValue(); }
    SVGAnimatedNumber& pathLengthAnimated() { return m_pathLength; }

protected:
    SVGGeometryElement(const QualifiedName&, Document&, UniqueRef<SVGPropertyRegistry>&&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

private:
    bool isSVGGeometryElement() const override { return true; }

    Ref<SVGAnimatedNumber> m_pathLength { SVGAnimatedNumber::create(this) };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SVGGeometryElement)
    static bool isType(const WebCore::SVGElement& element) { return element.isSVGGeometryElement(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* svgElement = dynamicDowncast<WebCore::SVGElement>(node);
        return svgElement && isType(*svgElement);
    }
SPECIALIZE_TYPE_TRAITS_END()
