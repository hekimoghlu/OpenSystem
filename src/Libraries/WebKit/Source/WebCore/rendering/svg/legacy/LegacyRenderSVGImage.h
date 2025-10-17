/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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

namespace WebCore {

class RenderImageResource;
class SVGImageElement;

class LegacyRenderSVGImage final : public LegacyRenderSVGModelObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGImage);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGImage);
public:
    LegacyRenderSVGImage(SVGImageElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGImage();

    SVGImageElement& imageElement() const;

    bool updateImageViewport();
    void setNeedsBoundariesUpdate() override { m_needsBoundariesUpdate = true; }
    void setNeedsTransformUpdate() override { m_needsTransformUpdate = true; }

    RenderImageResource& imageResource() { return *m_imageResource; }
    const RenderImageResource& imageResource() const { return *m_imageResource; }
    CheckedRef<RenderImageResource> checkedImageResource() const;

    // Note: Assumes the PaintInfo context has had all local transforms applied.
    void paintForeground(PaintInfo&);

private:
    void willBeDestroyed() override;

    void element() const = delete;

    ASCIILiteral renderName() const override { return "RenderSVGImage"_s; }
    bool canHaveChildren() const override { return false; }

    const AffineTransform& localToParentTransform() const override { return m_localTransform; }

    FloatRect calculateObjectBoundingBox() const;
    FloatRect objectBoundingBox() const override { return m_objectBoundingBox; }
    FloatRect strokeBoundingBox() const override { return m_objectBoundingBox; }
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const override { return m_repaintBoundingBox; }

    void addFocusRingRects(Vector<LayoutRect>&, const LayoutPoint& additionalOffset, const RenderLayerModelObject* paintContainer = 0) const override;

    void imageChanged(WrappedImagePtr, const IntRect* = nullptr) override;

    void layout() override;
    void paint(PaintInfo&, const LayoutPoint&) override;

    void invalidateBufferedForeground();

    bool nodeAtFloatPoint(const HitTestRequest&, HitTestResult&, const FloatPoint& pointInParent, HitTestAction) override;

    AffineTransform localTransform() const override { return m_localTransform; }

    bool m_needsBoundariesUpdate : 1;
    bool m_needsTransformUpdate : 1;
    AffineTransform m_localTransform;
    FloatRect m_objectBoundingBox;
    FloatRect m_repaintBoundingBox;
    std::unique_ptr<RenderImageResource> m_imageResource;
    RefPtr<ImageBuffer> m_bufferedForeground;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(LegacyRenderSVGImage, isLegacyRenderSVGImage())
