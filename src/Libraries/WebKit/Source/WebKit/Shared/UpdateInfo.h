/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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

#if USE(COORDINATED_GRAPHICS) || USE(TEXTURE_MAPPER)

#include <WebCore/IntRect.h>
#include <WebCore/ShareableBitmap.h>
#include <wtf/Noncopyable.h>
#include <wtf/Vector.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

struct UpdateInfo {
    // The size of the web view.
    WebCore::IntSize viewSize;
    float deviceScaleFactor { 0 };

    // The rect and delta to be scrolled.
    WebCore::IntRect scrollRect;
    WebCore::IntSize scrollOffset;
    
    // The bounds of the update rects.
    WebCore::IntRect updateRectBounds;

    // All the update rects, in view coordinates.
    Vector<WebCore::IntRect> updateRects;

    // The page scale factor used to render this update.
    float updateScaleFactor { 0 };

    // The handle of the shareable bitmap containing the updates. Will be null if there are no updates.
    std::optional<WebCore::ShareableBitmap::Handle> bitmapHandle;

    // The offset in the bitmap where the rendered contents are.
    WebCore::IntPoint bitmapOffset;
};

} // namespace WebKit

#endif
