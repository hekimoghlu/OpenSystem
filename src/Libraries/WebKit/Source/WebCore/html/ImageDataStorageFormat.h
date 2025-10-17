/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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

#include "PixelFormat.h"
#include <JavaScriptCore/TypedArrayType.h>

namespace WebCore {

enum class ImageDataStorageFormat : bool {
    Uint8,
    Float16,
};

constexpr PixelFormat toPixelFormat(ImageDataStorageFormat storageFormat)
{
    switch (storageFormat) {
    case ImageDataStorageFormat::Uint8:
        break;
    case ImageDataStorageFormat::Float16:
#if HAVE(HDR_SUPPORT)
        return PixelFormat::RGBA16F;
#else
        break;
#endif
    }
    return PixelFormat::RGBA8;
}

constexpr std::optional<ImageDataStorageFormat> toImageDataStorageFormat(JSC::TypedArrayType typedArrayType)
{
    switch (typedArrayType) {
    case JSC::TypeUint8Clamped: return ImageDataStorageFormat::Uint8;
    case JSC::TypeFloat16: return ImageDataStorageFormat::Float16;
    default: return std::nullopt;
    }
}

}
