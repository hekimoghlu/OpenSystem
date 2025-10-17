/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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

#import <wtf/FastMalloc.h>
#import <wtf/Ref.h>
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/RetainReleaseSwift.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakHashSet.h>
#import <wtf/WeakPtr.h>

struct WGPUTextureViewImpl {
};

namespace WebGPU {

class CommandEncoder;
class Device;
class Texture;

// https://gpuweb.github.io/gpuweb/#gputextureview
class TextureView : public RefCountedAndCanMakeWeakPtr<TextureView>, public WGPUTextureViewImpl {
    WTF_MAKE_TZONE_ALLOCATED(TextureView);
public:
    static Ref<TextureView> create(id<MTLTexture> texture, const WGPUTextureViewDescriptor& descriptor, const std::optional<WGPUExtent3D>& renderExtent, Texture& parentTexture, Device& device)
    {
        return adoptRef(*new TextureView(texture, descriptor, renderExtent, parentTexture, device));
    }
    static Ref<TextureView> createInvalid(Texture& texture, Device& device)
    {
        return adoptRef(*new TextureView(texture, device));
    }

    ~TextureView();

    void setLabel(String&&);

    bool isValid() const;

    id<MTLTexture> texture() const;
    id<MTLTexture> parentTexture() const;
    const WGPUTextureViewDescriptor& descriptor() const { return m_descriptor; }
    const std::optional<WGPUExtent3D>& renderExtent() const { return m_renderExtent; }

    Device& device() const { return m_device; }
    bool previouslyCleared() const;
    void setPreviouslyCleared(uint32_t mipLevel = 0, uint32_t slice = 0);
    uint32_t width() const;
    uint32_t height() const;
    uint32_t depthOrArrayLayers() const;
    WGPUTextureUsageFlags usage() const;
    uint32_t sampleCount() const;
    WGPUTextureFormat parentFormat() const;
    WGPUTextureFormat format() const;
    uint32_t parentMipLevelCount() const;
    uint32_t mipLevelCount() const;
    uint32_t baseMipLevel() const;
    WGPUTextureAspect aspect() const;
    uint32_t arrayLayerCount() const;
    uint32_t baseArrayLayer() const;
    WGPUTextureViewDimension dimension() const;
    bool isDestroyed() const;
    void destroy();
    void setCommandEncoder(CommandEncoder&) const;
    const Texture& apiParentTexture() const { return m_parentTexture; }
    Texture& apiParentTexture() { return m_parentTexture; }
    uint32_t parentRelativeSlice() const;
    uint32_t parentRelativeMipLevel() const;

private:
    TextureView(id<MTLTexture>, const WGPUTextureViewDescriptor&, const std::optional<WGPUExtent3D>&, Texture&, Device&);
    TextureView(Texture&, Device&);

    id<MTLTexture> m_texture { nil };

    const WGPUTextureViewDescriptor m_descriptor;
    const std::optional<WGPUExtent3D> m_renderExtent;

    const Ref<Device> m_device;
    Ref<Texture> m_parentTexture;
    mutable WeakHashSet<CommandEncoder> m_commandEncoders;
} SWIFT_SHARED_REFERENCE(refTextureView, derefTextureView);

} // namespace WebGPU

inline void refTextureView(WebGPU::TextureView* obj)
{
    ref(obj);
}

inline void derefTextureView(WebGPU::TextureView* obj)
{
    deref(obj);
}
