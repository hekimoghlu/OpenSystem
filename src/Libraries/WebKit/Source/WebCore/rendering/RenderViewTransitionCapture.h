/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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

#include "ImageBuffer.h"
#include "RenderReplaced.h"

namespace WebCore {

class RenderViewTransitionCapture final : public RenderReplaced {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderViewTransitionCapture);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderViewTransitionCapture);
public:
    RenderViewTransitionCapture(Type, Document&, RenderStyle&&);
    virtual ~RenderViewTransitionCapture();

    void setImage(RefPtr<ImageBuffer>);
    bool setCapturedSize(const LayoutSize&, const LayoutRect& overflowRect, const LayoutPoint& layerToLayoutOffset);

    void paintReplaced(PaintInfo&, const LayoutPoint& paintOffset) override;
    void intrinsicSizeChanged() override;

    void layout() override;

    FloatSize scale() const { return m_scale; }

    // Rect covered by the captured contents, in RenderLayer coordinates of the captured renderer
    LayoutRect captureOverflowRect() const { return m_overflowRect; }

    LayoutRect captureLocalOverflowRect() const { return m_localOverflowRect; }

    // Inset of the scaled capture from the visualOverflowRect()
    LayoutPoint captureContentInset() const;

    bool canUseExistingLayers() const { return !hasNonVisibleOverflow(); }

    bool paintsContent() const final;

    RefPtr<ImageBuffer> image() { return m_oldImage; }

private:
    ASCIILiteral renderName() const override { return "RenderViewTransitionCapture"_s; }
    String debugDescription() const override;

    void updateFromStyle() override;

    Node* nodeForHitTest() const override;

    RefPtr<ImageBuffer> m_oldImage;
    // The overflow rect that the captured image represents, in RenderLayer coordinates
    // of the captured renderer (see layerToLayoutOffset in ViewTransition.cpp).
    // The intrisic size subset of the image is stored as the intrinsic size of the RenderReplaced.
    LayoutRect m_overflowRect;
    // The offset between coordinates used by RenderLayer, and RenderObject
    // for the captured renderer
    LayoutPoint m_layerToLayoutOffset;
    // The overflow rect of the snapshot (replaced content), scaled and positioned
    // so that the intrinsic size of the image fits the replaced content rect.
    LayoutRect m_localOverflowRect;
    LayoutSize m_imageIntrinsicSize;
    // Scale factor between the intrinsic size and the replaced content rect size.
    FloatSize m_scale;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderViewTransitionCapture, isRenderViewTransitionCapture())
