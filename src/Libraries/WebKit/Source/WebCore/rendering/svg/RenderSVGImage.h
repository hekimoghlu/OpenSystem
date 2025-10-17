/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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

#include "RenderImageResource.h"
#include "RenderSVGModelObject.h"
#include "SVGBoundingBoxComputation.h"

namespace WebCore {

class SVGImageElement;

class RenderSVGImage final : public RenderSVGModelObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGImage);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGImage);
public:
    RenderSVGImage(SVGImageElement&, RenderStyle&&);
    virtual ~RenderSVGImage();

    SVGImageElement& imageElement() const;
    Ref<SVGImageElement> protectedImageElement() const;

    RenderImageResource& imageResource() { return *m_imageResource; }
    const RenderImageResource& imageResource() const { return *m_imageResource; }
    CheckedRef<RenderImageResource> checkedImageResource() const;

    bool updateImageViewport();

private:
    void willBeDestroyed() final;

    void element() const = delete;

    ASCIILiteral renderName() const final { return "RenderSVGImage"_s; }
    bool canHaveChildren() const final { return false; }

    FloatRect calculateObjectBoundingBox() const;
    FloatRect objectBoundingBox() const final { return m_objectBoundingBox; }
    FloatRect strokeBoundingBox() const final { return m_objectBoundingBox; }
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const final { return SVGBoundingBoxComputation::computeRepaintBoundingBox(*this); }

    void imageChanged(WrappedImagePtr, const IntRect* = nullptr) final;

    void layout() final;
    void paint(PaintInfo&, const LayoutPoint&) final;

    void paintForeground(PaintInfo&, const LayoutPoint&);
    ImageDrawResult paintIntoRect(PaintInfo&, const FloatRect&, const FloatRect&);

    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) final;

    void repaintOrMarkForLayout(const IntRect* = nullptr);
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final;
    bool bufferForeground(PaintInfo&, const LayoutPoint&);

    bool needsHasSVGTransformFlags() const final;

    void applyTransform(TransformationMatrix&, const RenderStyle&, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption>) const final;

    CachedImage* cachedImage() const { return imageResource().cachedImage(); }

    FloatRect m_objectBoundingBox;
    std::unique_ptr<RenderImageResource> m_imageResource;
    RefPtr<ImageBuffer> m_bufferedForeground;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGImage, isRenderSVGImage())
