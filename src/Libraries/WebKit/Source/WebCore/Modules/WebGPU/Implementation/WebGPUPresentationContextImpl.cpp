/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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
#include "WebGPUPresentationContextImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "NativeImage.h"
#include "WebGPUCanvasConfiguration.h"
#include "WebGPUConvertToBackingContext.h"
#include "WebGPUDeviceImpl.h"
#include "WebGPUTextureDescriptor.h"
#include "WebGPUTextureImpl.h"
#include <WebGPU/WebGPUExt.h>

#if PLATFORM(COCOA)
#include <wtf/MachSendRight.h>
#endif

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PresentationContextImpl);

PresentationContextImpl::PresentationContextImpl(WebGPUPtr<WGPUSurface>&& surface, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(surface))
    , m_convertToBackingContext(convertToBackingContext)
{
}

PresentationContextImpl::~PresentationContextImpl() = default;

void PresentationContextImpl::setSize(uint32_t width, uint32_t height)
{
    m_width = width;
    m_height = height;
}

static WGPUToneMappingMode convertToToneMappingMode(WebCore::WebGPU::CanvasToneMappingMode toneMappingMode)
{
    switch (toneMappingMode) {
    case WebCore::WebGPU::CanvasToneMappingMode::Standard:
        return WGPUToneMappingMode_Standard;
    case WebCore::WebGPU::CanvasToneMappingMode::Extended:
        return WGPUToneMappingMode_Extended;
    }

    ASSERT_NOT_REACHED();
    return WGPUToneMappingMode_Extended;
}

static WGPUCompositeAlphaMode convertToAlphaMode(WebCore::WebGPU::CanvasAlphaMode compositingAlphaMode)
{
    switch (compositingAlphaMode) {
    case WebCore::WebGPU::CanvasAlphaMode::Opaque:
        return WGPUCompositeAlphaMode_Opaque;
    case WebCore::WebGPU::CanvasAlphaMode::Premultiplied:
        return WGPUCompositeAlphaMode_Premultiplied;
    }

    ASSERT_NOT_REACHED();
    return WGPUCompositeAlphaMode_Premultiplied;
}

bool PresentationContextImpl::configure(const CanvasConfiguration& canvasConfiguration)
{
    m_swapChain = nullptr;

    m_format = canvasConfiguration.format;

    Ref convertToBackingContext = m_convertToBackingContext;

    WGPUSwapChainDescriptor backingDescriptor {
        .nextInChain = nullptr,
        .label = nullptr,
        .usage = convertToBackingContext->convertTextureUsageFlagsToBacking(canvasConfiguration.usage),
        .format = convertToBackingContext->convertToBacking(canvasConfiguration.format),
        .width = m_width,
        .height = m_height,
        .presentMode = WGPUPresentMode_Immediate,
        .viewFormats = canvasConfiguration.viewFormats.map([&](auto colorFormat) {
            return convertToBackingContext->convertToBacking(colorFormat);
        }),
        .colorSpace = canvasConfiguration.colorSpace == WebCore::WebGPU::PredefinedColorSpace::SRGB ? WGPUColorSpace::SRGB : WGPUColorSpace::DisplayP3,
        .toneMappingMode = convertToToneMappingMode(canvasConfiguration.toneMappingMode),
        .compositeAlphaMode = convertToAlphaMode(canvasConfiguration.compositingAlphaMode),
        .reportValidationErrors = canvasConfiguration.reportValidationErrors
    };

    m_swapChain = adoptWebGPU(wgpuDeviceCreateSwapChain(convertToBackingContext->convertToBacking(canvasConfiguration.protectedDevice().get()), m_backing.get(), &backingDescriptor));
    return true;
}

void PresentationContextImpl::unconfigure()
{
    if (!m_swapChain)
        return;

    m_swapChain = nullptr;
    
    m_format = TextureFormat::Bgra8unorm;
    m_width = 0;
    m_height = 0;
    m_swapChain = nullptr;
    m_currentTexture = nullptr;
}

RefPtr<Texture> PresentationContextImpl::getCurrentTexture(uint32_t frameIndex)
{
    if (!m_swapChain)
        return nullptr; // FIXME: This should return an invalid texture instead.

    if (!m_currentTexture) {
        auto texturePtr = wgpuSwapChainGetCurrentTexture(m_swapChain.get(), frameIndex);
        if (!texturePtr)
            return nullptr;

        m_currentTexture = TextureImpl::create(WebGPUPtr<WGPUTexture> { texturePtr }, m_format, TextureDimension::_2d, m_convertToBackingContext);
    }
    return m_currentTexture;
}

void PresentationContextImpl::present(uint32_t frameIndex, bool)
{
    if (auto* surface = m_swapChain.get())
        wgpuSwapChainPresent(surface, frameIndex);
    m_currentTexture = nullptr;
}

RefPtr<WebCore::NativeImage> PresentationContextImpl::getMetalTextureAsNativeImage(uint32_t bufferIndex, bool& isIOSurfaceSupportedFormat)
{
    if (auto* surface = m_swapChain.get())
        return WebCore::NativeImage::create(wgpuSwapChainGetTextureAsNativeImage(surface, bufferIndex, isIOSurfaceSupportedFormat));

    return nullptr;
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
