/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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

#include "GraphicsContextSwitcher.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ImageBuffer;

class ImageBufferContextSwitcher final : public GraphicsContextSwitcher {
    WTF_MAKE_TZONE_ALLOCATED(ImageBufferContextSwitcher);
public:
    ImageBufferContextSwitcher(GraphicsContext& destinationContext, const FloatRect& sourceImageRect, const DestinationColorSpace&, RefPtr<Filter>&& = nullptr, FilterResults* = nullptr);

private:
    GraphicsContext* drawingContext(GraphicsContext& destinationContext) const override;

    bool hasSourceImage() const override { return m_sourceImage; }

    void beginClipAndDrawSourceImage(GraphicsContext& destinationContext, const FloatRect& repaintRect, const FloatRect& clipRect) override;
    void endClipAndDrawSourceImage(GraphicsContext& destinationContext, const DestinationColorSpace&) override;

    void beginDrawSourceImage(GraphicsContext&, float = 1.f) override { }
    void endDrawSourceImage(GraphicsContext& destinationContext, const DestinationColorSpace&) override;

    RefPtr<ImageBuffer> m_sourceImage;
    FloatRect m_sourceImageRect;

    FilterResults* m_results { nullptr };
};

} // namespace WebCore
