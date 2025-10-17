/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#include "SVGGeometryElement.h"
#include "SVGNames.h"
#include "SVGPathByteStream.h"
#include "SVGPathSegImpl.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGPathSegList;
class SVGPoint;

class SVGPathElement final : public SVGGeometryElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGPathElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGPathElement);
public:
    static Ref<SVGPathElement> create(const QualifiedName&, Document&);

    static Ref<SVGPathSegClosePath> createSVGPathSegClosePath() { return SVGPathSegClosePath::create(); }
    static Ref<SVGPathSegMovetoAbs> createSVGPathSegMovetoAbs(float x, float y) { return SVGPathSegMovetoAbs::create(x, y); }
    static Ref<SVGPathSegMovetoRel> createSVGPathSegMovetoRel(float x, float y) { return SVGPathSegMovetoRel::create(x, y); }
    static Ref<SVGPathSegLinetoAbs> createSVGPathSegLinetoAbs(float x, float y) { return SVGPathSegLinetoAbs::create(x, y); }
    static Ref<SVGPathSegLinetoRel> createSVGPathSegLinetoRel(float x, float y) { return SVGPathSegLinetoRel::create(x, y); }
    static Ref<SVGPathSegCurvetoCubicAbs> createSVGPathSegCurvetoCubicAbs(float x, float y, float x1, float y1, float x2, float y2)
    {
        return SVGPathSegCurvetoCubicAbs::create(x, y, x1, y1, x2, y2);
    }
    static Ref<SVGPathSegCurvetoCubicRel> createSVGPathSegCurvetoCubicRel(float x, float y, float x1, float y1, float x2, float y2)
    {
        return SVGPathSegCurvetoCubicRel::create(x, y, x1, y1, x2, y2);
    }
    static Ref<SVGPathSegCurvetoQuadraticAbs> createSVGPathSegCurvetoQuadraticAbs(float x, float y, float x1, float y1)
    {
        return SVGPathSegCurvetoQuadraticAbs::create(x, y, x1, y1);
    }
    static Ref<SVGPathSegCurvetoQuadraticRel> createSVGPathSegCurvetoQuadraticRel(float x, float y, float x1, float y1)
    {
        return SVGPathSegCurvetoQuadraticRel::create(x, y, x1, y1);
    }
    static Ref<SVGPathSegArcAbs> createSVGPathSegArcAbs(float x, float y, float r1, float r2, float angle, bool largeArcFlag, bool sweepFlag)
    {
        return SVGPathSegArcAbs::create(x, y, r1, r2, angle, largeArcFlag, sweepFlag);
    }
    static Ref<SVGPathSegArcRel> createSVGPathSegArcRel(float x, float y, float r1, float r2, float angle, bool largeArcFlag, bool sweepFlag)
    {
        return SVGPathSegArcRel::create(x, y, r1, r2, angle, largeArcFlag, sweepFlag);
    }
    static Ref<SVGPathSegLinetoHorizontalAbs> createSVGPathSegLinetoHorizontalAbs(float x) { return SVGPathSegLinetoHorizontalAbs::create(x); }
    static Ref<SVGPathSegLinetoHorizontalRel> createSVGPathSegLinetoHorizontalRel(float x) { return SVGPathSegLinetoHorizontalRel::create(x); }
    static Ref<SVGPathSegLinetoVerticalAbs> createSVGPathSegLinetoVerticalAbs(float y) { return SVGPathSegLinetoVerticalAbs::create(y); }
    static Ref<SVGPathSegLinetoVerticalRel> createSVGPathSegLinetoVerticalRel(float y) { return SVGPathSegLinetoVerticalRel::create(y); }
    static Ref<SVGPathSegCurvetoCubicSmoothAbs> createSVGPathSegCurvetoCubicSmoothAbs(float x, float y, float x2, float y2)
    {
        return SVGPathSegCurvetoCubicSmoothAbs::create(x, y, x2, y2);
    }
    static Ref<SVGPathSegCurvetoCubicSmoothRel> createSVGPathSegCurvetoCubicSmoothRel(float x, float y, float x2, float y2)
    {
        return SVGPathSegCurvetoCubicSmoothRel::create(x, y, x2, y2);
    }
    static Ref<SVGPathSegCurvetoQuadraticSmoothAbs> createSVGPathSegCurvetoQuadraticSmoothAbs(float x, float y)
    {
        return SVGPathSegCurvetoQuadraticSmoothAbs::create(x, y);
    }
    static Ref<SVGPathSegCurvetoQuadraticSmoothRel> createSVGPathSegCurvetoQuadraticSmoothRel(float x, float y)
    {
        return SVGPathSegCurvetoQuadraticSmoothRel::create(x, y);
    }

    float getTotalLength() const final;
    ExceptionOr<Ref<SVGPoint>> getPointAtLength(float distance) const final;
    unsigned getPathSegAtLength(float distance) const;

    FloatRect getBBox(StyleUpdateStrategy = AllowStyleUpdate) final;

    Ref<SVGPathSegList>& pathSegList() { return m_pathSegList->baseVal(); }
    RefPtr<SVGPathSegList>& animatedPathSegList() { return m_pathSegList->animVal(); }

    const SVGPathByteStream& pathByteStream() const;
    Path path() const;
    size_t approximateMemoryCost() const final { return m_pathSegList->approximateMemoryCost(); }

    void pathDidChange();

    static void clearCache();

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGPathElement, SVGGeometryElement>;

private:
    SVGPathElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;

    bool isValid() const final { return SVGTests::isValid(); }
    bool supportsMarkers() const final { return true; }

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;

    Node::InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;

    void invalidateMPathDependencies();

    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) final;
    void collectExtraStyleForPresentationalHints(MutableStyleProperties&) override;
    void collectDPresentationalHint(MutableStyleProperties&);

    Ref<SVGAnimatedPathSegList> m_pathSegList { SVGAnimatedPathSegList::create(this) };
};

} // namespace WebCore
