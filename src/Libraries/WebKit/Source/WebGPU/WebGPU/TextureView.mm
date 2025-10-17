/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
#import "TextureView.h"

#import "APIConversions.h"
#import "CommandEncoder.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextureView);

TextureView::TextureView(id<MTLTexture> texture, const WGPUTextureViewDescriptor& descriptor, const std::optional<WGPUExtent3D>& renderExtent, Texture& parentTexture, Device& device)
    : m_texture(texture)
    , m_descriptor(descriptor)
    , m_renderExtent(renderExtent)
    , m_device(device)
    , m_parentTexture(parentTexture)
{
}

TextureView::TextureView(Texture& texture, Device& device)
    : m_descriptor { }
    , m_device(device)
    , m_parentTexture(texture)
{
}

TextureView::~TextureView() = default;

void TextureView::setLabel(String&& label)
{
    m_texture.label = label;
}

id<MTLTexture> TextureView::parentTexture() const
{
    return m_parentTexture->texture();
}

bool TextureView::previouslyCleared() const
{
    return Ref { m_parentTexture }->previouslyCleared(m_texture.parentRelativeLevel, m_texture.parentRelativeSlice);
}

void TextureView::setPreviouslyCleared(uint32_t mipLevel, uint32_t slice)
{
    Ref { m_parentTexture }->setPreviouslyCleared(m_texture.parentRelativeLevel + mipLevel, m_texture.parentRelativeSlice + slice);
}

uint32_t TextureView::parentRelativeMipLevel() const
{
    RELEASE_ASSERT(baseMipLevel() == m_texture.parentRelativeLevel);
    return m_texture.parentRelativeLevel;
}

uint32_t TextureView::parentRelativeSlice() const
{
    return m_texture.parentRelativeSlice;
}

uint32_t TextureView::width() const
{
    return Ref { m_parentTexture }->physicalMiplevelSpecificTextureExtent(baseMipLevel()).width;
}

uint32_t TextureView::height() const
{
    return Ref { m_parentTexture }->physicalMiplevelSpecificTextureExtent(baseMipLevel()).height;
}

uint32_t TextureView::depthOrArrayLayers() const
{
    return Ref { m_parentTexture }->physicalMiplevelSpecificTextureExtent(baseMipLevel()).depthOrArrayLayers;
}

WGPUTextureUsageFlags TextureView::usage() const
{
    return m_parentTexture->usage();
}

id<MTLTexture> TextureView::texture() const
{
    return isDestroyed() ? parentTexture() : m_texture;
}

uint32_t TextureView::sampleCount() const
{
    return m_parentTexture->sampleCount();
}

WGPUTextureFormat TextureView::parentFormat() const
{
    return m_parentTexture->format();
}

WGPUTextureFormat TextureView::format() const
{
    return m_descriptor.format;
}

uint32_t TextureView::parentMipLevelCount() const
{
    return m_parentTexture->mipLevelCount();
}

uint32_t TextureView::mipLevelCount() const
{
    return m_descriptor.mipLevelCount;
}

uint32_t TextureView::baseMipLevel() const
{
    return m_descriptor.baseMipLevel;
}

WGPUTextureAspect TextureView::aspect() const
{
    return m_descriptor.aspect;
}

uint32_t TextureView::arrayLayerCount() const
{
    return m_descriptor.arrayLayerCount;
}

uint32_t TextureView::baseArrayLayer() const
{
    return m_descriptor.baseArrayLayer;
}

WGPUTextureViewDimension TextureView::dimension() const
{
    return m_descriptor.dimension;
}

bool TextureView::isDestroyed() const
{
    return m_parentTexture->isDestroyed();
}

bool TextureView::isValid() const
{
    return m_texture;
}

void TextureView::destroy()
{
    m_texture = Ref { m_device }->placeholderTexture(format());
    if (!m_parentTexture->isCanvasBacking()) {
        for (Ref commandEncoder : m_commandEncoders)
            commandEncoder->makeSubmitInvalid();
    }

    m_commandEncoders.clear();
}

void TextureView::setCommandEncoder(CommandEncoder& commandEncoder) const
{
    CommandEncoder::trackEncoder(commandEncoder, m_commandEncoders);
    commandEncoder.addTexture(m_parentTexture);
    if (isDestroyed() && !m_parentTexture->isCanvasBacking())
        commandEncoder.makeSubmitInvalid();
}

} // namespace WebGPU

#pragma mark WGPU Stubs

void wgpuTextureViewReference(WGPUTextureView textureView)
{
    WebGPU::fromAPI(textureView).ref();
}

void wgpuTextureViewRelease(WGPUTextureView textureView)
{
    WebGPU::fromAPI(textureView).deref();
}

void wgpuTextureViewSetLabel(WGPUTextureView textureView, const char* label)
{
    WebGPU::protectedFromAPI(textureView)->setLabel(WebGPU::fromAPI(label));
}
