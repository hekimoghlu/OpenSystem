/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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

#include "BarcodeFormat.h"
#include "DOMRectReadOnly.h"
#include "DetectedBarcodeInterface.h"
#include "Point2D.h"
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct DetectedBarcode {
    ShapeDetection::DetectedBarcode convertToBacking() const
    {
        ASSERT(boundingBox);
        return {
            {
                static_cast<float>(boundingBox->x()),
                static_cast<float>(boundingBox->y()),
                static_cast<float>(boundingBox->width()),
                static_cast<float>(boundingBox->height()),
            },
            rawValue,
            WebCore::convertToBacking(format),
            cornerPoints.map([] (const auto& cornerPoint) {
                return cornerPoint.convertToBacking();
            }),
        };
    }

    RefPtr<DOMRectReadOnly> boundingBox;
    String rawValue;
    BarcodeFormat format { BarcodeFormat::Unknown };
    Vector<Point2D> cornerPoints;
};

inline DetectedBarcode convertFromBacking(const ShapeDetection::DetectedBarcode& detectedBarcode)
{
    return {
        DOMRectReadOnly::create(detectedBarcode.boundingBox.x(), detectedBarcode.boundingBox.y(), detectedBarcode.boundingBox.width(), detectedBarcode.boundingBox.height()),
        detectedBarcode.rawValue,
        convertFromBacking(detectedBarcode.format),
        detectedBarcode.cornerPoints.map([] (const auto& cornerPoint) {
            return Point2D { cornerPoint.x(), cornerPoint.y() };
        }),
    };
}

} // namespace WebCore
