/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
#import <wtf/FastMalloc.h>
#import <wtf/Ref.h>
#import <wtf/RefCounted.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/Vector.h>

struct WGPURenderBundleImpl {
};

@interface ResourceUsageAndRenderStage : NSObject
- (instancetype)initWithUsage:(MTLResourceUsage)usage renderStages:(MTLRenderStages)renderStages entryUsage:(OptionSet<WebGPU::BindGroupEntryUsage>)entryUsage binding:(uint32_t)binding resource:(WebGPU::BindGroupEntryUsageData::Resource)resource;

@property (nonatomic) MTLResourceUsage usage;
@property (nonatomic) MTLRenderStages renderStages;
@property (nonatomic) OptionSet<WebGPU::BindGroupEntryUsage> entryUsage;
@property (nonatomic) uint32_t binding;
@property (nonatomic) WebGPU::BindGroupEntryUsageData::Resource resource;
@end

@class RenderBundleICBWithResources;

namespace WebGPU {

class BindGroup;
class Buffer;
class CommandEncoder;
class Device;
class RenderBundleEncoder;
class RenderPassEncoder;
class RenderPipeline;
class TextureView;

// https://gpuweb.github.io/gpuweb/#gpurenderbundle
class RenderBundle : public WGPURenderBundleImpl, public RefCounted<RenderBundle> {
    WTF_MAKE_TZONE_ALLOCATED(RenderBundle);
public:
    using MinVertexCountsContainer = HashMap<uint64_t, IndexBufferAndIndexData, DefaultHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>>;
    using ResourcesContainer = NSMapTable<id<MTLResource>, ResourceUsageAndRenderStage*>;
    static Ref<RenderBundle> create(NSArray<RenderBundleICBWithResources*> *resources, RefPtr<WebGPU::RenderBundleEncoder> encoder, const WGPURenderBundleEncoderDescriptor& descriptor, uint64_t commandCount, bool makeSubmitInvalid, HashSet<RefPtr<const BindGroup>>&& bindGroups, Device& device)
    {
        return adoptRef(*new RenderBundle(resources, encoder, descriptor, commandCount, makeSubmitInvalid, WTFMove(bindGroups), device));
    }
    static Ref<RenderBundle> createInvalid(Device& device, NSString* errorString)
    {
        return adoptRef(*new RenderBundle(device, errorString));
    }

    ~RenderBundle();

    void setLabel(String&&);

    bool isValid() const;

    Device& device() const { return m_device; }
    NSArray<RenderBundleICBWithResources*> *renderBundlesResources() const { return m_renderBundlesResources; }

    void replayCommands(RenderPassEncoder&) const;
    void updateMinMaxDepths(float minDepth, float maxDepth);
    bool validateRenderPass(bool depthReadOnly, bool stencilReadOnly, const WGPURenderPassDescriptor&, const Vector<RefPtr<TextureView>>&, const RefPtr<TextureView>&) const;
    bool validatePipeline(const RenderPipeline*);
    uint64_t drawCount() const;
    NSString* lastError() const;
    bool requiresCommandReplay() const;
    bool makeSubmitInvalid() const;
    bool rebindSamplersIfNeeded() const;

private:
    RenderBundle(NSArray<RenderBundleICBWithResources*> *, RefPtr<RenderBundleEncoder>, const WGPURenderBundleEncoderDescriptor&, uint64_t, bool makeSubmitInvalid, HashSet<RefPtr<const BindGroup>>&&, Device&);
    RenderBundle(Device&, NSString*);

    const Ref<Device> m_device;
    RefPtr<RenderBundleEncoder> m_renderBundleEncoder;
    NSArray<RenderBundleICBWithResources*> *m_renderBundlesResources;
    WGPURenderBundleEncoderDescriptor m_descriptor;
    Vector<WGPUTextureFormat> m_descriptorColorFormats;
    HashSet<RefPtr<const BindGroup>> m_bindGroups;

    NSString* m_lastErrorString { nil };
    uint64_t m_commandCount { 0 };
    float m_minDepth { 0.f };
    float m_maxDepth { 1.f };
    bool m_makeSubmitInvalid { false };
};

} // namespace WebGPU
