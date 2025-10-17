/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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

namespace WebKit {

enum class GestureType : uint8_t {
    Loupe,
    OneFingerTap,
    TapAndAHalf,
    DoubleTap,
    OneFingerDoubleTap,
    OneFingerTripleTap,
    TwoFingerSingleTap,
    PhraseBoundary
};

enum class SelectionTouch : uint8_t {
    Started,
    Moved,
    Ended,
    EndedMovingForward,
    EndedMovingBackward,
    EndedNotMoving
};

enum class GestureRecognizerState : uint8_t {
    Possible,
    Began,
    Changed,
    Ended,
    Cancelled,
    Failed
};

enum class SheetAction : uint8_t {
    Copy,
    SaveImage,
    PauseAnimation,
    PlayAnimation,
#if ENABLE(SPATIAL_IMAGE_DETECTION)
    ViewSpatial
#endif
};

enum class SelectionFlags : uint8_t {
    WordIsNearTap = 1 << 0,
    SelectionFlipped = 1 << 1,
    PhraseBoundaryChanged = 1 << 2,
};

enum class RespectSelectionAnchor : bool { No, Yes };

enum class TextInteractionSource : uint8_t {
    Touch = 1 << 0,
    Mouse = 1 << 1,
};

enum class SelectionEndpoint : bool { Start, End };
enum class SelectionWasFlipped : bool { No, Yes };

} // namespace WebKit
