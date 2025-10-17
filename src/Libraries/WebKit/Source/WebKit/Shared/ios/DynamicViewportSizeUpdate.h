/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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

#import <WebCore/IntDegrees.h>

namespace WebKit {

typedef uint64_t DynamicViewportSizeUpdateID;

enum class DynamicViewportUpdateMode {
    NotResizing,
    ResizingWithAnimation,
    ResizingWithDocumentHidden,
};

struct DynamicViewportSizeUpdate {
    WebCore::FloatSize viewLayoutSize;
    WebCore::FloatSize minimumUnobscuredSize;
    WebCore::FloatSize maximumUnobscuredSize;
    WebCore::FloatRect exposedContentRect;
    WebCore::FloatRect unobscuredRect;
    WebCore::FloatRect unobscuredRectInScrollViewCoordinates;
    WebCore::FloatBoxExtent unobscuredSafeAreaInsets;
    double scale { 1 };
    WebCore::IntDegrees deviceOrientation { 0 };
    double minimumEffectiveDeviceWidth { 0 };
    DynamicViewportSizeUpdateID identifier;
};

} // namespace WebKit
