/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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

#if ENABLE(GPU_PROCESS)

#include "SwapBuffersDisplayRequirement.h"
#include <WebCore/FloatRect.h>
#include <WebCore/ImageBuffer.h>
#include <WebCore/Region.h>
#include <wtf/Vector.h>

namespace WebKit {

// An ImageBufferSet is a set of three ImageBuffers (front, back, secondary back),
// for the purpose of drawing successive (layer) frames.
// It handles picking an existing buffer to be the new front buffer, and then copying
// forward the previous pixels, clipping to the dirty area and clearing that subset.
// FIXME: This class should also have common code for buffer allocation, setting volatility,
// and accessing the GraphicsContext.
// Splitting out a virtual base class so that RemoteImageBufferProxy can implement the interface
// would also be nice.
class ImageBufferSet {
public:
    using PaintRectList = Vector<WebCore::FloatRect, 5>;

    // Tries to swap one of the existing back buffers to the new front buffer, if any are
    // not currently in-use.
    SwapBuffersDisplayRequirement swapBuffersForDisplay(bool hasEmptyDirtyRegion, bool supportsPartialRepaint);

    // Initializes the contents of the new front buffer using the previous
    // frames pixels(if applicable), clips to the dirty region, and clears the pixels
    // to be drawn (unless drawing will be opaque).
    void prepareBufferForDisplay(const WebCore::FloatRect& layerBounds, const WebCore::Region& dirtyRegion, const PaintRectList& paintingRects, bool requiresClearedPixels);


    static PaintRectList computePaintingRects(const WebCore::Region& dirtyRegion, float resolutionScale);

    void clearBuffers();

    RefPtr<WebCore::ImageBuffer> protectedFrontBuffer() { return m_frontBuffer; }
    RefPtr<WebCore::ImageBuffer> protectedBackBuffer() { return m_backBuffer; }
    RefPtr<WebCore::ImageBuffer> protectedSecondaryBackBuffer() { return m_secondaryBackBuffer; }

    RefPtr<WebCore::ImageBuffer> m_frontBuffer;
    RefPtr<WebCore::ImageBuffer> m_backBuffer;
    RefPtr<WebCore::ImageBuffer> m_secondaryBackBuffer;

    std::optional<WebCore::IntRect> m_previouslyPaintedRect;
    bool m_frontBufferIsCleared { false };
};

} // namespace WebKit

#endif
