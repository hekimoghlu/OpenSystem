/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#import "RenderBundle.h"

#import "APIConversions.h"
#import "BindGroup.h"
#import <wtf/TZoneMallocInlines.h>

@implementation ResourceUsageAndRenderStage
- (instancetype)initWithUsage:(MTLResourceUsage)usage renderStages:(MTLRenderStages)renderStages entryUsage:(OptionSet<WebGPU::BindGroupEntryUsage>)entryUsage binding:(uint32_t)binding resource:(WebGPU::BindGroupEntryUsageData::Resource)resource
{
    if (!(self = [super init]))
        return nil;

    _usage = usage;
    _renderStages = renderStages;
    _entryUsage = entryUsage;
    _binding = binding;
    _resource = resource;

    return self;
}
@end

namespace WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderBundle);

RenderBundle::RenderBundle(NSArray<RenderBundleICBWithResources*> *resources, RefPtr<RenderBundleEncoder> encoder, const WGPURenderBundleEncoderDescriptor& descriptor, uint64_t commandCount, bool makeSubmitInvalid, HashSet<RefPtr<const BindGroup>>&& bindGroups, Device& device)
    : m_device(device)
    , m_renderBundleEncoder(encoder)
    , m_renderBundlesResources(resources)
    , m_descriptor(descriptor)
    , m_descriptorColorFormats(descriptor.colorFormatsSpan())
    , m_bindGroups(bindGroups)
    , m_commandCount(commandCount)
    , m_makeSubmitInvalid(makeSubmitInvalid)
{
    if (m_descriptorColorFormats.size())
        m_descriptor.colorFormats = &m_descriptorColorFormats[0];

    ASSERT(m_renderBundleEncoder || m_renderBundlesResources.count);
}

RenderBundle::RenderBundle(Device& device, NSString* errorString)
    : m_device(device)
    , m_lastErrorString(errorString)
{
}

RenderBundle::~RenderBundle() = default;

bool RenderBundle::isValid() const
{
    return m_renderBundleEncoder || m_renderBundlesResources.count;
}

void RenderBundle::setLabel(String&& label)
{
    m_renderBundlesResources.firstObject.indirectCommandBuffer.label = label;
}

void RenderBundle::replayCommands(RenderPassEncoder& renderPassEncoder) const
{
    if (RefPtr renderBundleEncoder = m_renderBundleEncoder)
        renderBundleEncoder->replayCommands(renderPassEncoder);
}

bool RenderBundle::requiresCommandReplay() const
{
    return !!m_renderBundleEncoder.get();
}

void RenderBundle::updateMinMaxDepths(float minDepth, float maxDepth)
{
    if (m_minDepth == minDepth && m_maxDepth == maxDepth)
        return;

    m_minDepth = minDepth;
    m_maxDepth = maxDepth;
    std::array<float, 2> twoFloats = { m_minDepth, m_maxDepth };
    for (RenderBundleICBWithResources* icb in m_renderBundlesResources)
        m_device->protectedQueue()->writeBuffer(icb.fragmentDynamicOffsetsBuffer, 0, asWritableBytes(std::span(twoFloats)));
}

uint64_t RenderBundle::drawCount() const
{
    return m_commandCount;
}

bool RenderBundle::validateRenderPass(bool depthReadOnly, bool stencilReadOnly, const WGPURenderPassDescriptor& descriptor, const Vector<RefPtr<TextureView>>& colorAttachmentViews, const RefPtr<TextureView>& depthStencilView) const
{
    if (depthReadOnly && !m_descriptor.depthReadOnly)
        return false;

    if (stencilReadOnly && !m_descriptor.stencilReadOnly)
        return false;

    if (m_descriptor.colorFormatCount != descriptor.colorAttachmentCount)
        return false;

    auto descriptorColorFormats = m_descriptor.colorFormatsSpan();

    uint32_t defaultRasterSampleCount = 0;
    for (size_t i = 0, colorFormatCount = std::max(descriptor.colorAttachmentCount, m_descriptor.colorFormatCount); i < colorFormatCount; ++i) {
        auto descriptorColorFormat = i < descriptorColorFormats.size() ? descriptorColorFormats[i] : WGPUTextureFormat_Undefined;
        if (i >= descriptor.colorAttachmentCount) {
            if (descriptorColorFormat == WGPUTextureFormat_Undefined)
                continue;
            return false;
        }
        auto attachmentView = colorAttachmentViews[i];
        if (!attachmentView) {
            if (descriptorColorFormat == WGPUTextureFormat_Undefined)
                continue;
            return false;
        }
        if (descriptorColorFormat != attachmentView->format())
            return false;
        defaultRasterSampleCount = attachmentView->sampleCount();
    }

    if (descriptor.depthStencilAttachment) {
        if (!depthStencilView) {
            if (m_descriptor.depthStencilFormat != WGPUTextureFormat_Undefined)
                return false;
        } else {
            auto& texture = *depthStencilView.get();
            if (texture.format() != m_descriptor.depthStencilFormat)
                return false;
            defaultRasterSampleCount = texture.sampleCount();
        }
    } else if (m_descriptor.depthStencilFormat != WGPUTextureFormat_Undefined)
        return false;

    if (m_descriptor.sampleCount != defaultRasterSampleCount)
        return false;

    return true;
}

bool RenderBundle::validatePipeline(const RenderPipeline*)
{
    return true;
}

NSString* RenderBundle::lastError() const
{
    return m_lastErrorString;
}

bool RenderBundle::makeSubmitInvalid() const
{
    return m_makeSubmitInvalid;
}

bool RenderBundle::rebindSamplersIfNeeded() const
{
    bool result = true;
    for (RefPtr bindGroup : m_bindGroups)
        result = bindGroup->rebindSamplersIfNeeded() && result;

    return result;
}

} // namespace WebGPU

#pragma mark WGPU Stubs

void wgpuRenderBundleReference(WGPURenderBundle renderBundle)
{
    WebGPU::fromAPI(renderBundle).ref();
}

void wgpuRenderBundleRelease(WGPURenderBundle renderBundle)
{
    WebGPU::fromAPI(renderBundle).deref();
}

void wgpuRenderBundleSetLabel(WGPURenderBundle renderBundle, const char* label)
{
    WebGPU::protectedFromAPI(renderBundle)->setLabel(WebGPU::fromAPI(label));
}
