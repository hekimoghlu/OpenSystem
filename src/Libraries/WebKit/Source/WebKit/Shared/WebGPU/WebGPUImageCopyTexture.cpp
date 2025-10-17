/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
#include "config.h"
#include "WebGPUImageCopyTexture.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUImageCopyTexture.h>
#include <WebCore/WebGPUTexture.h>

namespace WebKit::WebGPU {

std::optional<ImageCopyTexture> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::ImageCopyTexture& imageCopyTexture)
{
    auto texture = convertToBacking(imageCopyTexture.protectedTexture().get());

    std::optional<Origin3D> origin;
    if (imageCopyTexture.origin) {
        origin = convertToBacking(*imageCopyTexture.origin);
        if (!origin)
            return std::nullopt;
    }

    return { { texture, imageCopyTexture.mipLevel, WTFMove(origin), imageCopyTexture.aspect } };
}

std::optional<WebCore::WebGPU::ImageCopyTexture> ConvertFromBackingContext::convertFromBacking(const ImageCopyTexture& imageCopyTexture)
{
    WeakPtr texture = convertTextureFromBacking(imageCopyTexture.texture);
    if (!texture)
        return std::nullopt;

    std::optional<WebCore::WebGPU::Origin3D> origin;
    if (imageCopyTexture.origin) {
        origin = convertFromBacking(*imageCopyTexture.origin);
        if (!origin)
            return std::nullopt;
    }

    return { { *texture, imageCopyTexture.mipLevel, WTFMove(origin), imageCopyTexture.aspect } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
