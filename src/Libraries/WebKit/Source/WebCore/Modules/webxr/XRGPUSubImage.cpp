/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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
#include "XRGPUSubImage.h"

#if ENABLE(WEBXR_LAYERS)

#include "GPUDevice.h"
#include "GPUTextureDescriptor.h"
#include "GPUTextureFormat.h"
#include "WebXRViewport.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static constexpr auto gpuSubImageWidth = 1920;
static constexpr auto gpuSubImageHeight = 1824;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(XRGPUSubImage);

static auto makeTextureViewDescriptor(WebGPU::XREye eye)
{
    return GPUTextureViewDescriptor {
        .format = std::nullopt,
        .dimension = GPUTextureViewDimension::_2d,
        .aspect = GPUTextureAspect::All,
        .baseMipLevel = 0,
        .mipLevelCount = std::nullopt,
        .baseArrayLayer = (eye == WebGPU::XREye::Right ? 1u : 0u),
        .arrayLayerCount = 1u,
    };
}

XRGPUSubImage::XRGPUSubImage(Ref<WebGPU::XRSubImage>&& backing, WebGPU::XREye eye, GPUDevice& device)
    : m_backing(WTFMove(backing))
    , m_device(device)
    , m_descriptor(makeTextureViewDescriptor(eye))
    , m_viewport(WebXRViewport::create(IntRect { 0, 0, gpuSubImageWidth, gpuSubImageHeight }))
{
}

static GPUTextureDescriptor textureDescriptor(GPUTextureFormat format)
{
    return GPUTextureDescriptor {
        { "canvas backing"_s },
        GPUExtent3DDict { gpuSubImageWidth, gpuSubImageHeight, 1 },
        1,
        1,
        GPUTextureDimension::_2d,
        format,
        GPUTextureUsage::RENDER_ATTACHMENT,
        { }
    };
}

ExceptionOr<Ref<GPUTexture>> XRGPUSubImage::colorTexture()
{
    RefPtr texture = m_backing->colorTexture();
    if (!texture)
        return Exception { ExceptionCode::InvalidStateError };

    return GPUTexture::create(texture.releaseNonNull(), textureDescriptor(GPUTextureFormat::Bgra8unormSRGB), m_device);
}

RefPtr<GPUTexture> XRGPUSubImage::depthStencilTexture()
{
    RefPtr texture = m_backing->depthStencilTexture();
    if (!texture)
        return nullptr;

    return GPUTexture::create(texture.releaseNonNull(), textureDescriptor(GPUTextureFormat::Depth24plus), m_device);
}

RefPtr<GPUTexture> XRGPUSubImage::motionVectorTexture()
{
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

const GPUTextureViewDescriptor& XRGPUSubImage::getViewDescriptor() const
{
    return m_descriptor;
}

const WebXRViewport& XRGPUSubImage::viewport() const
{
    return m_viewport;
}

} // namespace WebCore

#endif // ENABLE(WEBXR_LAYERS)
