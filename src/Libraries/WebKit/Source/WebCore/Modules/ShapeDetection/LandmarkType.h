/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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

#include "LandmarkTypeInterface.h"
#include <cstdint>

namespace WebCore {

enum class LandmarkType : uint8_t {
    Mouth,
    Eye,
    Nose,
};

inline ShapeDetection::LandmarkType convertToBacking(LandmarkType landmarkType)
{
    switch (landmarkType) {
    case LandmarkType::Mouth:
        return ShapeDetection::LandmarkType::Mouth;
    case LandmarkType::Eye:
        return ShapeDetection::LandmarkType::Eye;
    case LandmarkType::Nose:
        return ShapeDetection::LandmarkType::Nose;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

inline LandmarkType convertFromBacking(ShapeDetection::LandmarkType landmarkType)
{
    switch (landmarkType) {
    case ShapeDetection::LandmarkType::Mouth:
        return LandmarkType::Mouth;
    case ShapeDetection::LandmarkType::Eye:
        return LandmarkType::Eye;
    case ShapeDetection::LandmarkType::Nose:
        return LandmarkType::Nose;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WebCore
