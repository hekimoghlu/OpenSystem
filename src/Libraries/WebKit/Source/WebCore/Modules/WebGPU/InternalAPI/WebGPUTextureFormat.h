/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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

#include <cstdint>

namespace WebCore::WebGPU {

enum class TextureFormat : uint8_t {
    // 8-bit formats
    R8unorm,
    R8snorm,
    R8uint,
    R8sint,

    // 16-bit formats
    R16uint,
    R16sint,
    R16float,
    Rg8unorm,
    Rg8snorm,
    Rg8uint,
    Rg8sint,

    // 32-bit formats
    R32uint,
    R32sint,
    R32float,
    Rg16uint,
    Rg16sint,
    Rg16float,
    Rgba8unorm,
    Rgba8unormSRGB,
    Rgba8snorm,
    Rgba8uint,
    Rgba8sint,
    Bgra8unorm,
    Bgra8unormSRGB,
    // Packed 32-bit formats
    Rgb9e5ufloat,
    Rgb10a2uint,
    Rgb10a2unorm,
    Rg11b10ufloat,

    // 64-bit formats
    Rg32uint,
    Rg32sint,
    Rg32float,
    Rgba16uint,
    Rgba16sint,
    Rgba16float,

    // 128-bit formats
    Rgba32uint,
    Rgba32sint,
    Rgba32float,

    // Depth/stencil formats
    Stencil8,
    Depth16unorm,
    Depth24plus,
    Depth24plusStencil8,
    Depth32float,

    // depth32float-stencil8 feature
    Depth32floatStencil8,

    // BC compressed formats usable if texture-compression-bc is both
    // supported by the device/user agent and enabled in requestDevice.
    Bc1RgbaUnorm,
    Bc1RgbaUnormSRGB,
    Bc2RgbaUnorm,
    Bc2RgbaUnormSRGB,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSRGB,
    Bc4RUnorm,
    Bc4RSnorm,
    Bc5RgUnorm,
    Bc5RgSnorm,
    Bc6hRgbUfloat,
    Bc6hRgbFloat,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSRGB,

    // ETC2 compressed formats usable if texture-compression-etc2 is both
    // supported by the device/user agent and enabled in requestDevice.
    Etc2Rgb8unorm,
    Etc2Rgb8unormSRGB,
    Etc2Rgb8a1unorm,
    Etc2Rgb8a1unormSRGB,
    Etc2Rgba8unorm,
    Etc2Rgba8unormSRGB,
    EacR11unorm,
    EacR11snorm,
    EacRg11unorm,
    EacRg11snorm,

    // ASTC compressed formats usable if texture-compression-astc is both
    // supported by the device/user agent and enabled in requestDevice.
    Astc4x4Unorm,
    Astc4x4UnormSRGB,
    Astc5x4Unorm,
    Astc5x4UnormSRGB,
    Astc5x5Unorm,
    Astc5x5UnormSRGB,
    Astc6x5Unorm,
    Astc6x5UnormSRGB,
    Astc6x6Unorm,
    Astc6x6UnormSRGB,
    Astc8x5Unorm,
    Astc8x5UnormSRGB,
    Astc8x6Unorm,
    Astc8x6UnormSRGB,
    Astc8x8Unorm,
    Astc8x8UnormSRGB,
    Astc10x5Unorm,
    Astc10x5UnormSRGB,
    Astc10x6Unorm,
    Astc10x6UnormSRGB,
    Astc10x8Unorm,
    Astc10x8UnormSRGB,
    Astc10x10Unorm,
    Astc10x10UnormSRGB,
    Astc12x10Unorm,
    Astc12x10UnormSRGB,
    Astc12x12Unorm,
    Astc12x12UnormSRGB,
};

} // namespace WebCore::WebGPU
