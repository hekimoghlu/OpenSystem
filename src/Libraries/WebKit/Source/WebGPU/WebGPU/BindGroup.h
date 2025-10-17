/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

#import "BindableResource.h"
#import "ShaderStage.h"
#import <wtf/EnumeratedArray.h>
#import <wtf/FastMalloc.h>
#import <wtf/Ref.h>
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/Vector.h>
#import <wtf/WeakPtr.h>

struct WGPUBindGroupImpl {
};

namespace WebGPU {

class BindGroupLayout;
class Device;
class ExternalTexture;
class Sampler;

struct ExternalTextureIndices {
    NSUInteger containerIndex { NSNotFound };
    NSUInteger resourceIndex { NSNotFound };
    NSUInteger argumentBufferIndex { NSNotFound };
};

// https://gpuweb.github.io/gpuweb/#gpubindgroup
class BindGroup : public RefCountedAndCanMakeWeakPtr<BindGroup>, public WGPUBindGroupImpl {
    WTF_MAKE_TZONE_ALLOCATED(BindGroup);
public:
    template <typename T>
    using ShaderStageArray = EnumeratedArray<ShaderStage, T, ShaderStage::Compute>;
    using SamplersContainer = HashMap<RefPtr<Sampler>, ShaderStageArray<std::optional<uint32_t>>>;
    struct BufferAndType {
        WGPUBufferBindingType type;
        uint64_t bindingSize;
        uint64_t bufferSize;
        uint32_t bindingIndex;
    };
    using DynamicBuffersContainer = Vector<BufferAndType>;

    static constexpr MTLRenderStages MTLRenderStageCompute = static_cast<MTLRenderStages>(0);
    static constexpr MTLRenderStages MTLRenderStageUndefined = static_cast<MTLRenderStages>(MTLRenderStageFragment + 1);
    static Ref<BindGroup> create(id<MTLBuffer> vertexArgumentBuffer, id<MTLBuffer> fragmentArgumentBuffer, id<MTLBuffer> computeArgumentBuffer, Vector<BindableResources>&& resources, const BindGroupLayout& bindGroupLayout, DynamicBuffersContainer&& dynamicBuffers, SamplersContainer&& samplers, ShaderStageArray<ExternalTextureIndices>&& externalTextureIndices, Device& device)
    {
        return adoptRef(*new BindGroup(vertexArgumentBuffer, fragmentArgumentBuffer, computeArgumentBuffer, WTFMove(resources), bindGroupLayout, WTFMove(dynamicBuffers), WTFMove(samplers), WTFMove(externalTextureIndices), device));
    }
    static Ref<BindGroup> createInvalid(Device& device)
    {
        return adoptRef(*new BindGroup(device));
    }

    ~BindGroup();

    void setLabel(String&&);

    bool isValid() const;

    id<MTLBuffer> vertexArgumentBuffer() const { return m_vertexArgumentBuffer; }
    id<MTLBuffer> fragmentArgumentBuffer() const { return m_fragmentArgumentBuffer; }
    id<MTLBuffer> computeArgumentBuffer() const { return m_computeArgumentBuffer; }

    const Vector<BindableResources>& resources() const { return m_resources; }

    Device& device() const { return m_device; }
    Ref<Device> protectedDevice() const { return m_device; }
    static bool allowedUsage(const OptionSet<BindGroupEntryUsage>&);
    static NSString* usageName(const OptionSet<BindGroupEntryUsage>&);
    static uint64_t makeEntryMapKey(uint32_t baseMipLevel, uint32_t baseArrayLayer, WGPUTextureAspect);

    const BindGroupLayout* bindGroupLayout() const { return m_bindGroupLayout.get(); }
    RefPtr<const BindGroupLayout> protectedBindGroupLayout() const { return m_bindGroupLayout; }

    const BufferAndType* dynamicBuffer(uint32_t) const;
    uint32_t dynamicOffset(uint32_t bindingIndex, const Vector<uint32_t>*) const;
    bool rebindSamplersIfNeeded() const;
    bool updateExternalTextures(ExternalTexture&);
    bool makeSubmitInvalid(ShaderStage, const BindGroupLayout*) const;
    const SamplersContainer& samplers() const { return m_samplers; }

private:
    BindGroup(id<MTLBuffer> vertexArgumentBuffer, id<MTLBuffer> fragmentArgumentBuffer, id<MTLBuffer> computeArgumentBuffer, Vector<BindableResources>&&, const BindGroupLayout&, DynamicBuffersContainer&&, SamplersContainer&&, ShaderStageArray<ExternalTextureIndices>&&, Device&);
    BindGroup(Device&);

    const id<MTLBuffer> m_vertexArgumentBuffer { nil };
    const id<MTLBuffer> m_fragmentArgumentBuffer { nil };
    const id<MTLBuffer> m_computeArgumentBuffer { nil };

    const Ref<Device> m_device;
    Vector<BindableResources> m_resources;
    RefPtr<const BindGroupLayout> m_bindGroupLayout;
    DynamicBuffersContainer m_dynamicBuffers;
    HashMap<uint32_t, uint32_t, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>> m_dynamicOffsetsIndices;
    SamplersContainer m_samplers;
    ShaderStageArray<ExternalTextureIndices> m_externalTextureIndices;
};

} // namespace WebGPU
