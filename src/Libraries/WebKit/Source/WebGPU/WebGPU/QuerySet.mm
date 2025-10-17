/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#import "QuerySet.h"

#import "APIConversions.h"
#import "Buffer.h"
#import "CommandEncoder.h"
#import "Device.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebGPU {

Ref<QuerySet> Device::createQuerySet(const WGPUQuerySetDescriptor& descriptor)
{
    auto count = descriptor.count;
    constexpr auto maxCountAllowed = 4096;
    if (descriptor.nextInChain || count > maxCountAllowed || !isValid()) {
        generateAValidationError("GPUQuerySetDescriptor.count must be <= 4096"_s);
        return QuerySet::createInvalid(*this);
    }

    const char* label = descriptor.label;
    auto type = descriptor.type;

    switch (type) {
    case WGPUQueryType_Timestamp: {
#if !PLATFORM(WATCHOS)
        MTLCounterSampleBufferDescriptor* sampleBufferDesc = [MTLCounterSampleBufferDescriptor new];
        sampleBufferDesc.sampleCount = count;
        sampleBufferDesc.storageMode = MTLStorageModeShared;
        sampleBufferDesc.counterSet = m_capabilities.baseCapabilities.timestampCounterSet;

        NSError* error = nil;
        id<MTLCounterSampleBuffer> buffer = [m_device newCounterSampleBufferWithDescriptor:sampleBufferDesc error:&error];
        if (error)
            return QuerySet::createInvalid(*this);
        return QuerySet::create(buffer, count, type, *this);
#else
        return QuerySet::createInvalid(*this);
#endif
    } case WGPUQueryType_Occlusion: {
        auto buffer = safeCreateBuffer(sizeof(uint64_t) * count, MTLStorageModePrivate);
        buffer.label = fromAPI(label);
        return QuerySet::create(buffer, count, type, *this);
    }
    case WGPUQueryType_Force32:
        return QuerySet::createInvalid(*this);
    }
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(QuerySet);

QuerySet::QuerySet(id<MTLBuffer> buffer, uint32_t count, WGPUQueryType type, Device& device)
    : m_device(device)
    , m_visibilityBuffer(buffer)
    , m_count(count)
    , m_type(type)
{
    RELEASE_ASSERT(m_type != WGPUQueryType_Force32);
}

QuerySet::QuerySet(id<MTLCounterSampleBuffer> buffer, uint32_t count, WGPUQueryType type, Device& device)
    : m_device(device)
    , m_timestampBuffer(buffer)
    , m_count(count)
    , m_type(type)
{
    RELEASE_ASSERT(m_type != WGPUQueryType_Force32);
}

QuerySet::QuerySet(Device& device)
    : m_device(device)
    , m_type(WGPUQueryType_Force32)
{
}

QuerySet::~QuerySet() = default;

bool QuerySet::isValid() const
{
    return isDestroyed() || m_visibilityBuffer || m_timestampBuffer;
}

bool QuerySet::isDestroyed() const
{
    return m_destroyed;
}

void QuerySet::destroy()
{
    m_destroyed = true;
    // https://gpuweb.github.io/gpuweb/#dom-gpuqueryset-destroy
    m_visibilityBuffer = nil;
    m_timestampBuffer = nil;
    for (Ref commandEncoder : m_commandEncoders)
        commandEncoder->makeSubmitInvalid();

    m_commandEncoders.clear();
}

void QuerySet::setLabel(String&& label)
{
    m_visibilityBuffer.label = label;
    // MTLCounterSampleBuffer's label property is read-only.
}

void QuerySet::setOverrideLocation(QuerySet&, uint32_t, uint32_t)
{
}

void QuerySet::setCommandEncoder(CommandEncoder& commandEncoder) const
{
    CommandEncoder::trackEncoder(commandEncoder, m_commandEncoders);
    commandEncoder.addBuffer(m_visibilityBuffer);
    commandEncoder.addBuffer(m_timestampBuffer);
    if (isDestroyed())
        commandEncoder.makeSubmitInvalid();
}

} // namespace WebGPU

#pragma mark WGPU Stubs

void wgpuQuerySetReference(WGPUQuerySet querySet)
{
    WebGPU::fromAPI(querySet).ref();
}

void wgpuQuerySetRelease(WGPUQuerySet querySet)
{
    WebGPU::fromAPI(querySet).deref();
}

void wgpuQuerySetDestroy(WGPUQuerySet querySet)
{
    WebGPU::protectedFromAPI(querySet)->destroy();
}

void wgpuQuerySetSetLabel(WGPUQuerySet querySet, const char* label)
{
    WebGPU::protectedFromAPI(querySet)->setLabel(WebGPU::fromAPI(label));
}

uint32_t wgpuQuerySetGetCount(WGPUQuerySet querySet)
{
    return WebGPU::protectedFromAPI(querySet)->count();
}

WGPUQueryType wgpuQuerySetGetType(WGPUQuerySet querySet)
{
    return WebGPU::protectedFromAPI(querySet)->type();
}
