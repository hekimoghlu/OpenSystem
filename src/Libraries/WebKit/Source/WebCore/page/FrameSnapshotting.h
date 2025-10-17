/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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

#include "Color.h"
#include "DestinationColorSpace.h"
#include "ImageBufferPixelFormat.h"
#include "SimpleRange.h"
#include <memory>
#include <wtf/OptionSet.h>

namespace WebCore {

class FloatRect;
class IntRect;
class ImageBuffer;
class LocalFrame;
class Node;

enum class SnapshotFlags : uint16_t {
    ExcludeSelectionHighlighting = 1 << 0,
    PaintSelectionOnly = 1 << 1,
    InViewCoordinates = 1 << 2,
    ForceBlackText = 1 << 3,
    PaintSelectionAndBackgroundsOnly = 1 << 4,
    PaintEverythingExcludingSelection = 1 << 5,
    PaintWithIntegralScaleFactor = 1 << 6,
    Shareable = 1 << 7,
    Accelerated = 1 << 8,
    ExcludeReplacedContent = 1 << 9,
    PaintWith3xBaseScale = 1 << 10,
};

struct SnapshotOptions {
    OptionSet<SnapshotFlags> flags;
    ImageBufferPixelFormat pixelFormat;
    DestinationColorSpace colorSpace;
};

WEBCORE_EXPORT RefPtr<ImageBuffer> snapshotFrameRect(LocalFrame&, const IntRect&, SnapshotOptions&&);
RefPtr<ImageBuffer> snapshotFrameRectWithClip(LocalFrame&, const IntRect&, const Vector<FloatRect>& clipRects, SnapshotOptions&&);
RefPtr<ImageBuffer> snapshotNode(LocalFrame&, Node&, SnapshotOptions&&);
WEBCORE_EXPORT RefPtr<ImageBuffer> snapshotSelection(LocalFrame&, SnapshotOptions&&);

Color estimatedBackgroundColorForRange(const SimpleRange&, const LocalFrame&);

} // namespace WebCore
