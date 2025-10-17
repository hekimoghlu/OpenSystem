/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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

#if ENABLE(VIDEO)

#include "HTMLVideoElement.h"
#include "RenderMedia.h"

namespace WebCore {

class RenderVideo final : public RenderMedia {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderVideo);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderVideo);
public:
    RenderVideo(HTMLVideoElement&, RenderStyle&&);
    virtual ~RenderVideo();

    WEBCORE_EXPORT HTMLVideoElement& videoElement() const;

    IntRect videoBox() const;
    WEBCORE_EXPORT IntRect videoBoxInRootView() const;

    static IntSize defaultSize();

    bool supportsAcceleratedRendering() const;
    void acceleratedRenderingStateChanged();

    bool requiresImmediateCompositing() const;

    bool shouldDisplayVideo() const;
    bool failedToLoadPosterImage() const;

    void updateFromElement() final;
    bool hasVideoMetadata() const;
    bool hasPosterFrameSize() const;
    bool hasDefaultObjectSize() const;

private:
    void willBeDestroyed() override;
    void mediaElement() const = delete;

    void intrinsicSizeChanged() final;
    LayoutSize calculateIntrinsicSizeInternal();
    LayoutSize calculateIntrinsicSize();
    bool updateIntrinsicSize();

    void imageChanged(WrappedImagePtr, const IntRect*) final;

    ASCIILiteral renderName() const final { return "RenderVideo"_s; }

    bool requiresLayer() const final { return true; }

    void paintReplaced(PaintInfo&, const LayoutPoint&) final;

    void layout() final;
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) final;

    void visibleInViewportStateChanged() final;

    LayoutUnit computeReplacedLogicalWidth(ShouldComputePreferred  = ShouldComputePreferred::ComputeActual) const final;
    LayoutUnit minimumReplacedHeight() const final;

    bool updatePlayer();

    bool foregroundIsKnownToBeOpaqueInRect(const LayoutRect& localRect, unsigned maxDepthToTest) const final;
    void invalidateLineLayout();

    LayoutSize m_cachedImageSize;
};

inline RenderVideo* HTMLVideoElement::renderer() const
{
    return downcast<RenderVideo>(HTMLMediaElement::renderer());
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderVideo, isRenderVideo())

#endif // ENABLE(VIDEO)
