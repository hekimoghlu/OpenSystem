/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
#import "PresentationContextCoreAnimation.h"

#import "APIConversions.h"
#import "Texture.h"
#import "TextureView.h"
#import <wtf/TZoneMallocInlines.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebGPU {

static CAMetalLayer *layerFromSurfaceDescriptor(const WGPUSurfaceDescriptor& descriptor)
{
    ASSERT(descriptor.nextInChain->sType == WGPUSType_SurfaceDescriptorFromMetalLayer);
    ASSERT(!descriptor.nextInChain->next);
    const auto& metalDescriptor = *reinterpret_cast<const WGPUSurfaceDescriptorFromMetalLayer*>(descriptor.nextInChain);
    CAMetalLayer *layer = bridge_id_cast(metalDescriptor.layer);
    return layer;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(PresentationContextCoreAnimation);

PresentationContextCoreAnimation::PresentationContextCoreAnimation(const WGPUSurfaceDescriptor& descriptor)
    : m_layer(layerFromSurfaceDescriptor(descriptor))
{
}

PresentationContextCoreAnimation::~PresentationContextCoreAnimation() = default;

void PresentationContextCoreAnimation::configure(Device& device, const WGPUSwapChainDescriptor& descriptor)
{
    m_configuration = std::nullopt;

    if (descriptor.nextInChain)
        return;

    switch (descriptor.format) {
    case WGPUTextureFormat_BGRA8Unorm:
    case WGPUTextureFormat_BGRA8UnormSrgb:
    case WGPUTextureFormat_RGBA16Float:
        break;
    case WGPUTextureFormat_RGB10A2Unorm:
        if (device.baseCapabilities().canPresentRGB10A2PixelFormats)
            break;
        return;
    default:
        return;
    }

    m_configuration = Configuration(descriptor.width, descriptor.height, descriptor.usage, fromAPI(descriptor.label), descriptor.format, device);

    m_layer.pixelFormat = Texture::pixelFormat(descriptor.format);
    if (descriptor.usage == WGPUTextureUsage_RenderAttachment)
        m_layer.framebufferOnly = YES;
    m_layer.drawableSize = CGSizeMake(descriptor.width, descriptor.height);
    m_layer.device = device.device();
}

void PresentationContextCoreAnimation::unconfigure()
{
    m_configuration = std::nullopt;
}

auto PresentationContextCoreAnimation::Configuration::generateCurrentFrameState(CAMetalLayer *layer) -> Configuration::FrameState
{
    auto label = this->label.utf8();

    id<CAMetalDrawable> currentDrawable = [layer nextDrawable];
    id<MTLTexture> backingTexture = currentDrawable.texture;

    WGPUTextureDescriptor textureDescriptor {
        nullptr,
        label.data(),
        usage,
        WGPUTextureDimension_2D, {
            width,
            height,
            1,
        },
        format,
        1,
        1,
        0,
        nullptr,
    };
    auto texture = Texture::create(backingTexture, textureDescriptor, { format }, device);

    WGPUTextureViewDescriptor textureViewDescriptor {
        nullptr,
        label.data(),
        format,
        WGPUTextureViewDimension_2D,
        0,
        1,
        0,
        1,
        WGPUTextureAspect_All,
    };

    auto textureView = TextureView::create(backingTexture, textureViewDescriptor, { { width, height, 1 } }, texture, device);
    return { currentDrawable, texture.ptr(), textureView.ptr() };
}

void PresentationContextCoreAnimation::present(uint32_t)
{
    if (!m_configuration)
        return;

    if (!m_configuration->currentFrameState)
        m_configuration->currentFrameState = m_configuration->generateCurrentFrameState(m_layer);

    [m_configuration->currentFrameState->currentDrawable present];

    m_configuration->currentFrameState = std::nullopt;
}

Texture* PresentationContextCoreAnimation::getCurrentTexture(uint32_t)
{
    if (!m_configuration)
        return nullptr;

    if (!m_configuration->currentFrameState)
        m_configuration->currentFrameState = m_configuration->generateCurrentFrameState(m_layer);

    return m_configuration->currentFrameState->currentTexture.get();
}

TextureView* PresentationContextCoreAnimation::getCurrentTextureView()
{
    if (!m_configuration)
        return nullptr;

    if (!m_configuration->currentFrameState)
        m_configuration->currentFrameState = m_configuration->generateCurrentFrameState(m_layer);

    return m_configuration->currentFrameState->currentTextureView.get();
}

} // namespace WebGPU
