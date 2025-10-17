/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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

#include "AffineTransform.h"
#include "FloatSize.h"
#include "IntSize.h"

namespace WebCore {

struct ImageResolution {
    static constexpr unsigned DefaultResolution = 72;

    // Since this class is to be used mainly with EXIF,
    // this range intentionally matches the resolution values from the EXIF spec.
    // See JEITA CP-3451, page 18. http://www.exif.org/Exif2-2.PDF

    enum ResolutionUnit {
        None = 1,
        Inches = 2,
        Centimeters = 3
    };

    struct ResolutionMetadata {
        FloatSize preferredSize;
        FloatSize resolution;
        ResolutionUnit resolutionUnit;
    };

    static std::optional<IntSize> densityCorrectedSize(const FloatSize& sourceSize, const ResolutionMetadata&);
};

} // namespace WebCore
