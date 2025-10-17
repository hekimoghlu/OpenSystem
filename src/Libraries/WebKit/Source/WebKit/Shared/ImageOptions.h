/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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

#include <wtf/OptionSet.h>

namespace WebKit {

enum class ImageOption : uint8_t {
    Shareable = 1 << 0,
    // Makes local in process buffer
    Local = 1 << 1,
};

using ImageOptions = OptionSet<ImageOption>;

enum class SnapshotOption : uint16_t {
    Shareable = 1 << 0,
    ExcludeSelectionHighlighting = 1 << 1,
    InViewCoordinates = 1 << 2,
    PaintSelectionRectangle = 1 << 3,
    ExcludeDeviceScaleFactor = 1 << 5,
    ForceBlackText = 1 << 6,
    ForceWhiteText = 1 << 7,
    Printing = 1 << 8,
    UseScreenColorSpace = 1 << 9,
    VisibleContentRect = 1 << 10,
    FullContentRect = 1 << 11,
    TransparentBackground = 1 << 12,
};

using SnapshotOptions = OptionSet<SnapshotOption>;

inline ImageOptions snapshotOptionsToImageOptions(SnapshotOptions snapshotOptions)
{
    if (snapshotOptions.contains(SnapshotOption::Shareable))
        return ImageOption::Shareable;
    return { };
}

} // namespace WebKit
