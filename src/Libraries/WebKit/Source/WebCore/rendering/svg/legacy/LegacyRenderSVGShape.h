/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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

#include "AffineTransform.h"
#include "FloatRect.h"
#include "LegacyRenderSVGModelObject.h"
#include "SVGMarkerData.h"
#include <memory>
#include <wtf/Vector.h>

namespace WebCore {

class FloatPoint;
class GraphicsContextStateSaver;
class SVGGraphicsElement;

class LegacyRenderSVGShape : public LegacyRenderSVGModelObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGShape);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGShape);
public:
    friend FloatRect SVGRenderSupport::calculateApproximateStrokeBoundingBox(const RenderElement&);

    enum class ShapeType : uint8_t {
        Empty,
        Path,
        Line,
        Rectangle,
        RoundedRectangle,
        Ellipse,
        Circle,
    };

    enum PointCoordinateSpace {
        GlobalCoordinateSpace,
        LocalCoordinateSpace
    };
    LegacyRenderSVGShape(Type, SVGGraphicsElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGShape();

    inline SVGGraphicsElement& graphicsElement() const;
    inline Ref<SVGGraphicsElement> protectedGraphicsElement() const;

    void setNeedsShapeUpdate() { m_needsShapeUpdate = true; }
    void setNeedsBoundariesUpdate() final { m_needsBoundariesUpdate = true; }
    void setNeedsTransformUpdate() final { m_needsTransformUpdate = true; }
    virtual void fillShape(GraphicsContext&) const;
    virtual void strokeShape(GraphicsContext&) const;
    virtual bool isRenderingDisabled() const = 0;

    bool isPointInFill(const FloatPoint&);
    bool isPointInStroke(const FloatPoint&);

    float getTotalLength() const;
    FloatPoint getPointAtLength(float distance) const;

    bool hasPath() const { return m_path.get(); }
    Path& path() const
    {
        ASSERT(m_path);
        return *m_path;
    }
    void clearPath() { m_path = nullptr; }

    ShapeType shapeType() const { return m_shapeType; }

protected:
    void element() const = delete;

    Path& ensurePath();

    virtual void updateShapeFromElement() = 0;
    virtual bool isEmpty() const;
    virtual bool shapeDependentStrokeContains(const FloatPoint&, PointCoordinateSpace = GlobalCoordinateSpace);
    virtual bool shapeDependentFillContains(const FloatPoint&, const WindRule) const;
    float strokeWidth() const;

    inline bool hasNonScalingStroke() const;
    AffineTransform nonScalingStrokeTransform() const;
    Path* nonScalingStrokePath(const Path*, const AffineTransform&) const;

    virtual FloatRect adjustStrokeBoundingBoxForMarkersAndZeroLengthLinecaps(RepaintRectCalculation, FloatRect strokeBoundingBox) const { return strokeBoundingBox; }

    FloatRect strokeBoundingBox() const final;
    FloatRect approximateStrokeBoundingBox() const;

    bool fillRequiresClip() const { return m_fillRequiresClip; }

private:
    // Hit-detection separated for the fill and the stroke
    bool fillContains(const FloatPoint&, bool requiresFill = true, const WindRule fillRule = WindRule::NonZero);
    bool strokeContains(const FloatPoint&, bool requiresStroke = true);

    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const final;
    const AffineTransform& localToParentTransform() const final { return m_localTransform; }
    AffineTransform localTransform() const final { return m_localTransform; }

    bool canHaveChildren() const final { return false; }
    ASCIILiteral renderName() const override { return "RenderSVGShape"_s; }

    void layout() final;
    void paint(PaintInfo&, const LayoutPoint&) final;
    void addFocusRingRects(Vector<LayoutRect>&, const LayoutPoint& additionalOffset, const RenderLayerModelObject* paintContainer = 0) const final;

    bool nodeAtFloatPoint(const HitTestRequest&, HitTestResult&, const FloatPoint& pointInParent, HitTestAction) final;

    FloatRect objectBoundingBox() const final { return m_fillBoundingBox; }
    FloatRect calculateStrokeBoundingBox() const;
    void updateRepaintBoundingBox();

    bool setupNonScalingStrokeContext(AffineTransform&, GraphicsContextStateSaver&);

    FloatRect calculateApproximateStrokeBoundingBox() const;
    
    std::unique_ptr<Path> createPath() const;

    void fillShape(const RenderStyle&, GraphicsContext&);
    void strokeShapeInternal(const RenderStyle&, GraphicsContext&);
    void strokeShape(const RenderStyle&, GraphicsContext&);
    void fillStrokeMarkers(PaintInfo&);

    virtual void drawMarkers(PaintInfo&) { }

protected:
    FloatRect m_fillBoundingBox;
    mutable Markable<FloatRect, FloatRect::MarkableTraits> m_strokeBoundingBox;
    mutable Markable<FloatRect, FloatRect::MarkableTraits> m_approximateStrokeBoundingBox;
private:
    FloatRect m_repaintBoundingBox;

    bool m_needsBoundariesUpdate : 1;
    bool m_needsShapeUpdate : 1;
    bool m_needsTransformUpdate : 1;
    bool m_fillRequiresClip : 1 { true };
protected:
    ShapeType m_shapeType : 3 { ShapeType::Empty };
private:
    AffineTransform m_localTransform;
    std::unique_ptr<Path> m_path;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(LegacyRenderSVGShape, isLegacyRenderSVGShape())
