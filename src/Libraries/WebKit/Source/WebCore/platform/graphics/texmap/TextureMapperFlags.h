/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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

#if USE(TEXTURE_MAPPER)

namespace WebCore {

enum class TextureMapperFlags : uint16_t {
    ShouldBlend = 1 << 0,
    ShouldFlipTexture = 1 << 1,
    ShouldAntialias = 1 << 2,
    ShouldRotateTexture90 = 1 << 3,
    ShouldRotateTexture180 = 1 << 4,
    ShouldRotateTexture270 = 1 << 5,
    ShouldConvertTextureBGRAToRGBA = 1 << 6,
    ShouldConvertTextureARGBToRGBA = 1 << 7,
    ShouldNotBlend = 1 << 8,
    ShouldUseExternalOESTextureRect = 1 << 9,
    ShouldPremultiply = 1 << 10
};

} // namespace WebCore

#endif // USE(TEXTURE_MAPPER)
