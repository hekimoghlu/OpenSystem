/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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

#include "GeneratedImage.h"
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class LegacyRenderSVGResourceContainer;
class RenderElement;
class RenderSVGResourceContainer;

class SVGResourceImage final : public GeneratedImage {
public:
    static Ref<SVGResourceImage> create(RenderSVGResourceContainer&, const URL& reresolvedURL);
    static Ref<SVGResourceImage> create(LegacyRenderSVGResourceContainer&, const URL& reresolvedURL);

private:
    SVGResourceImage(RenderSVGResourceContainer&, const URL& reresolvedURL);
    SVGResourceImage(LegacyRenderSVGResourceContainer&, const URL& reresolvedURL);

    ImageDrawResult draw(GraphicsContext&, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions = { }) final;
    void drawPattern(GraphicsContext&, const FloatRect& destRect, const FloatRect& srcRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions = { }) final;

    bool isSVGResourceImage() const final { return true; }

    void dump(WTF::TextStream&) const final;

    SingleThreadWeakPtr<RenderSVGResourceContainer> m_renderResource;
    SingleThreadWeakPtr<LegacyRenderSVGResourceContainer> m_legacyRenderResource;
    URL m_reresolvedURL;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_IMAGE(SVGResourceImage)
