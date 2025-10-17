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
#import "config.h"
#import "XRSubImage.h"

#import "APIConversions.h"
#import "Device.h"
#import "Texture.h"

#import <wtf/CheckedArithmetic.h>
#import <wtf/StdLibExtras.h>

namespace WebGPU {

XRSubImage::XRSubImage(bool, Device& device)
    : m_device(device)
{
}

XRSubImage::XRSubImage(Device& device)
    : m_device(device)
{
}

XRSubImage::~XRSubImage() = default;

Ref<XRSubImage> Device::createXRSubImage()
{
    if (!isValid())
        return XRSubImage::createInvalid(*this);

    return XRSubImage::create(*this);
}

void XRSubImage::setLabel(String&&)
{
}

bool XRSubImage::isValid() const
{
    return true;
}

void XRSubImage::update(id<MTLTexture> colorTexture, id<MTLTexture> depthTexture, size_t currentTextureIndex, const std::pair<id<MTLSharedEvent>, uint64_t>& sharedEvent)
{
    RefPtr device = m_device.get();
    if (!device)
        return;

    m_currentTextureIndex = currentTextureIndex;
    RefPtr texture = this->colorTexture();
    if (!texture || texture->texture() != colorTexture) {
        auto colorFormat = WGPUTextureFormat_BGRA8UnormSrgb;
        WGPUTextureDescriptor colorTextureDescriptor = {
            .nextInChain = nullptr,
            .label = "color texture",
            .usage = WGPUTextureUsage_RenderAttachment,
            .dimension = WGPUTextureDimension_2D,
            .size = {
                .width = static_cast<uint32_t>(colorTexture.width),
                .height = static_cast<uint32_t>(colorTexture.height),
                .depthOrArrayLayers = static_cast<uint32_t>(colorTexture.arrayLength),
            },
            .format = colorFormat,
            .mipLevelCount = 1,
            .sampleCount = static_cast<uint32_t>(colorTexture.sampleCount),
            .viewFormatCount = 1,
            .viewFormats = &colorFormat,
        };
        auto newTexture = Texture::create(colorTexture, colorTextureDescriptor, { colorFormat }, *device);
        newTexture->updateCompletionEvent(sharedEvent);
        m_colorTextures.set(currentTextureIndex, newTexture.ptr());
    } else
        texture->updateCompletionEvent(sharedEvent);

    if (texture = this->depthTexture(); !texture || texture->texture() != depthTexture) {
        auto depthFormat = WGPUTextureFormat_Depth24PlusStencil8;
        WGPUTextureDescriptor depthTextureDescriptor = {
            .nextInChain = nullptr,
            .label = "depth texture",
            .usage = WGPUTextureUsage_RenderAttachment,
            .dimension = WGPUTextureDimension_2D,
            .size = {
                .width = static_cast<uint32_t>(depthTexture.width),
                .height = static_cast<uint32_t>(depthTexture.height),
                .depthOrArrayLayers = static_cast<uint32_t>(depthTexture.arrayLength),
            },
            .format = depthFormat,
            .mipLevelCount = 1,
            .sampleCount = static_cast<uint32_t>(depthTexture.sampleCount),
            .viewFormatCount = 1,
            .viewFormats = &depthFormat,
        };
        m_depthTextures.set(currentTextureIndex, Texture::create(depthTexture, depthTextureDescriptor, { depthFormat }, *device));
    }
}

Texture* XRSubImage::colorTexture()
{
    if (auto it = m_colorTextures.find(m_currentTextureIndex); it != m_colorTextures.end())
        return it->value.get();
    return nullptr;
}

Texture* XRSubImage::depthTexture()
{
    if (auto it = m_depthTextures.find(m_currentTextureIndex); it != m_depthTextures.end())
        return it->value.get();
    return nullptr;
}

RefPtr<XRSubImage> XRBinding::getViewSubImage(XRProjectionLayer& projectionLayer)
{
    return protectedDevice()->getXRViewSubImage(projectionLayer);
}

} // namespace WebGPU

#pragma mark WGPU Stubs

void wgpuXRSubImageReference(WGPUXRSubImage subImage)
{
    WebGPU::fromAPI(subImage).ref();
}

void wgpuXRSubImageRelease(WGPUXRSubImage subImage)
{
    WebGPU::fromAPI(subImage).deref();
}

WGPUTexture wgpuXRSubImageGetColorTexture(WGPUXRSubImage subImage)
{
    return WebGPU::protectedFromAPI(subImage)->colorTexture();
}

WGPUTexture wgpuXRSubImageGetDepthStencilTexture(WGPUXRSubImage subImage)
{
    return WebGPU::protectedFromAPI(subImage)->depthTexture();
}
