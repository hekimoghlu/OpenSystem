/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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
#include "WebGPUXRSubImageImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUConvertToBackingContext.h"
#include "WebGPUDevice.h"
#include "WebGPUTextureDimension.h"
#include "WebGPUTextureImpl.h"

namespace WebCore::WebGPU {

XRSubImageImpl::XRSubImageImpl(WebGPUPtr<WGPUXRSubImage>&& backing, ConvertToBackingContext& convertToBackingContext)
    : m_backing(backing)
    , m_convertToBackingContext(convertToBackingContext)
{
}

XRSubImageImpl::~XRSubImageImpl() = default;

RefPtr<Texture> XRSubImageImpl::colorTexture()
{
    auto texturePtr = wgpuXRSubImageGetColorTexture(m_backing.get());
    if (!texturePtr)
        return nullptr;

    return TextureImpl::create(WebGPUPtr<WGPUTexture> { texturePtr }, TextureFormat::Bgra8unormSRGB, TextureDimension::_2d, m_convertToBackingContext);
}

RefPtr<Texture> XRSubImageImpl::depthStencilTexture()
{
    auto texturePtr = wgpuXRSubImageGetDepthStencilTexture(m_backing.get());
    if (!texturePtr)
        return nullptr;

    return TextureImpl::create(WebGPUPtr<WGPUTexture> { texturePtr }, TextureFormat::Depth24plusStencil8, TextureDimension::_2d, m_convertToBackingContext);
}

RefPtr<Texture> XRSubImageImpl::motionVectorTexture()
{
    return nullptr;
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
