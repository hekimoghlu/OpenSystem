/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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

#include "DOMRectReadOnly.h"
#include "DetectedFaceInterface.h"
#include "Landmark.h"
#include <optional>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

struct Landmark;

struct DetectedFace {
    ShapeDetection::DetectedFace convertToBacking() const
    {
        ASSERT(boundingBox);
        return {
            {
                static_cast<float>(boundingBox->x()),
                static_cast<float>(boundingBox->y()),
                static_cast<float>(boundingBox->width()),
                static_cast<float>(boundingBox->height()),
            },
            landmarks ? std::optional { landmarks->map([] (const auto& landmark) {
                return landmark.convertToBacking();
            }) } : std::nullopt,
        };
    }

    RefPtr<DOMRectReadOnly> boundingBox;
    std::optional<Vector<Landmark>> landmarks;
};

inline DetectedFace convertFromBacking(const ShapeDetection::DetectedFace& detectedFace)
{
    return {
        DOMRectReadOnly::create(detectedFace.boundingBox.x(), detectedFace.boundingBox.y(), detectedFace.boundingBox.width(), detectedFace.boundingBox.height()),
        detectedFace.landmarks ? std::optional { detectedFace.landmarks->map([] (const auto& landmark) {
            return convertFromBacking(landmark);
        }) } : std::nullopt,
    };
}

} // namespace WebCore
