/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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

#include "BarcodeFormatInterface.h"
#include <cstdint>

namespace WebCore {

enum class BarcodeFormat : uint8_t {
    Aztec,
    Code_128,
    Code_39,
    Code_93,
    Codabar,
    Data_matrix,
    Ean_13,
    Ean_8,
    Itf,
    Pdf417,
    Qr_code,
    Unknown,
    Upc_a,
    Upc_e,
};

inline ShapeDetection::BarcodeFormat convertToBacking(BarcodeFormat barcodeFormat)
{
    switch (barcodeFormat) {
    case BarcodeFormat::Aztec:
        return ShapeDetection::BarcodeFormat::Aztec;
    case BarcodeFormat::Code_128:
        return ShapeDetection::BarcodeFormat::Code_128;
    case BarcodeFormat::Code_39:
        return ShapeDetection::BarcodeFormat::Code_39;
    case BarcodeFormat::Code_93:
        return ShapeDetection::BarcodeFormat::Code_93;
    case BarcodeFormat::Codabar:
        return ShapeDetection::BarcodeFormat::Codabar;
    case BarcodeFormat::Data_matrix:
        return ShapeDetection::BarcodeFormat::Data_matrix;
    case BarcodeFormat::Ean_13:
        return ShapeDetection::BarcodeFormat::Ean_13;
    case BarcodeFormat::Ean_8:
        return ShapeDetection::BarcodeFormat::Ean_8;
    case BarcodeFormat::Itf:
        return ShapeDetection::BarcodeFormat::Itf;
    case BarcodeFormat::Pdf417:
        return ShapeDetection::BarcodeFormat::Pdf417;
    case BarcodeFormat::Qr_code:
        return ShapeDetection::BarcodeFormat::Qr_code;
    case BarcodeFormat::Unknown:
        return ShapeDetection::BarcodeFormat::Unknown;
    case BarcodeFormat::Upc_a:
        return ShapeDetection::BarcodeFormat::Upc_a;
    case BarcodeFormat::Upc_e:
        return ShapeDetection::BarcodeFormat::Upc_e;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

inline BarcodeFormat convertFromBacking(ShapeDetection::BarcodeFormat barcodeFormat)
{
    switch (barcodeFormat) {
    case ShapeDetection::BarcodeFormat::Aztec:
        return BarcodeFormat::Aztec;
    case ShapeDetection::BarcodeFormat::Code_128:
        return BarcodeFormat::Code_128;
    case ShapeDetection::BarcodeFormat::Code_39:
        return BarcodeFormat::Code_39;
    case ShapeDetection::BarcodeFormat::Code_93:
        return BarcodeFormat::Code_93;
    case ShapeDetection::BarcodeFormat::Codabar:
        return BarcodeFormat::Codabar;
    case ShapeDetection::BarcodeFormat::Data_matrix:
        return BarcodeFormat::Data_matrix;
    case ShapeDetection::BarcodeFormat::Ean_13:
        return BarcodeFormat::Ean_13;
    case ShapeDetection::BarcodeFormat::Ean_8:
        return BarcodeFormat::Ean_8;
    case ShapeDetection::BarcodeFormat::Itf:
        return BarcodeFormat::Itf;
    case ShapeDetection::BarcodeFormat::Pdf417:
        return BarcodeFormat::Pdf417;
    case ShapeDetection::BarcodeFormat::Qr_code:
        return BarcodeFormat::Qr_code;
    case ShapeDetection::BarcodeFormat::Unknown:
        return BarcodeFormat::Unknown;
    case ShapeDetection::BarcodeFormat::Upc_a:
        return BarcodeFormat::Upc_a;
    case ShapeDetection::BarcodeFormat::Upc_e:
        return BarcodeFormat::Upc_e;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WebCore
