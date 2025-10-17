/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
#include "RenderSVGModelObject.h"
#include "SVGBoundingBoxComputation.h"
#include "SVGGraphicsElement.h"
#include "SVGMarkerData.h"
#include <memory>
#include <wtf/Vector.h>

namespace WebCore {

class FloatPoint;
class GraphicsContextStateSaver;
class SVGGraphicsElement;

class RenderSVGShape : public RenderSVGModelObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGShape);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGShape);
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
    RenderSVGShape(Type, SVGGraphicsElement&, RenderStyle&&);
    virtual ~RenderSVGShape();

    inline SVGGraphicsElement& graphicsElement() const;
    inline Ref<SVGGraphicsElement> protectedGraphicsElement() const;

    void setNeedsShapeUpdate() { m_needsShapeUpdate = true; }

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

    FloatRect objectBoundingBox() const final { return m_fillBoundingBox; }
    FloatRect strokeBoundingBox() const final;
    FloatRect approximateStrokeBoundingBox() const;
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const final { return SVGBoundingBoxComputation::computeRepaintBoundingBox(*this); }

    bool needsHasSVGTransformFlags() const final;

    void applyTransform(TransformationMatrix&, const RenderStyle&, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption>) const final;

    AffineTransform nonScalingStrokeTransform() const;

protected:
    void element() const = delete;

    Path& ensurePath();

    virtual void updateShapeFromElement() = 0;
    virtual bool isEmpty() const;
    virtual bool shapeDependentStrokeContains(const FloatPoint&, PointCoordinateSpace = GlobalCoordinateSpace);
    virtual bool shapeDependentFillContains(const FloatPoint&, const WindRule) const;
    float strokeWidth() const;

    inline bool hasNonScalingStroke() const;
    Path* nonScalingStrokePath(const Path*, const AffineTransform&) const;

    virtual FloatRect adjustStrokeBoundingBoxForZeroLengthLinecaps(RepaintRectCalculation, FloatRect strokeBoundingBox) const { return strokeBoundingBox; }

private:
    // Hit-detection separated for the fill and the stroke
    bool fillContains(const FloatPoint&, bool requiresFill = true, const WindRule fillRule = WindRule::NonZero);
    bool strokeContains(const FloatPoint&, bool requiresStroke = true);

    bool canHaveChildren() const final { return false; }
    ASCIILiteral renderName() const override { return "RenderSVGShape"_s; }

    void layout() final;
    void paint(PaintInfo&, const LayoutPoint&) final;

    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) override;

    FloatRect calculateStrokeBoundingBox() const;

    bool setupNonScalingStrokeContext(AffineTransform&, GraphicsContextStateSaver&);

    
    std::unique_ptr<Path> createPath() const;

    void fillShape(const RenderStyle&, GraphicsContext&);
    void strokeShape(const RenderStyle&, GraphicsContext&);
    void fillStrokeMarkers(PaintInfo&);
    virtual void drawMarkers(PaintInfo&) { }

    void styleWillChange(StyleDifference, const RenderStyle& newStyle) override;

    FloatRect calculateApproximateStrokeBoundingBox() const;

protected:
    FloatRect m_fillBoundingBox;
    mutable Markable<FloatRect, FloatRect::MarkableTraits> m_strokeBoundingBox;
    mutable Markable<FloatRect, FloatRect::MarkableTraits> m_approximateStrokeBoundingBox;
private:
    bool m_needsShapeUpdate { true };
protected:
    ShapeType m_shapeType : 3 { ShapeType::Empty };
private:
    std::unique_ptr<Path> m_path;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGShape, isRenderSVGShape())
