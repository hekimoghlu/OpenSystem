/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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

#import "WebGPU.h"
#import "WebGPUExt.h"
#import <optional>
#import <wtf/FastMalloc.h>
#import <wtf/Ref.h>
#import <wtf/RefCounted.h>
#import <wtf/RetainReleaseSwift.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/Vector.h>
#import <wtf/WeakHashSet.h>
#import <wtf/WeakPtr.h>

struct WGPUQuerySetImpl {
};

namespace WebGPU {

class Buffer;
class CommandEncoder;
class Device;

// https://gpuweb.github.io/gpuweb/#gpuqueryset
class QuerySet : public WGPUQuerySetImpl, public RefCounted<QuerySet> {
    WTF_MAKE_TZONE_ALLOCATED(QuerySet);
public:
    static Ref<QuerySet> create(id<MTLBuffer> visibilityBuffer, uint32_t count, WGPUQueryType type, Device& device)
    {
        return adoptRef(*new QuerySet(visibilityBuffer, count, type, device));
    }
    static Ref<QuerySet> create(id<MTLCounterSampleBuffer> counterSampleBuffer, uint32_t count, WGPUQueryType type, Device& device)
    {
        return adoptRef(*new QuerySet(counterSampleBuffer, count, type, device));
    }
    static Ref<QuerySet> createInvalid(Device& device)
    {
        return adoptRef(*new QuerySet(device));
    }

    ~QuerySet();

    void destroy();
    void setLabel(String&&);

    bool isValid() const;

    void setOverrideLocation(QuerySet& otherQuerySet, uint32_t beginningOfPassIndex, uint32_t endOfPassIndex);

    Device& device() const { return m_device; }
    uint32_t count() const { return m_count; }
    WGPUQueryType type() const { return m_type; }
    id<MTLBuffer> visibilityBuffer() const { return m_visibilityBuffer; }
    id<MTLCounterSampleBuffer> counterSampleBuffer() const { return m_timestampBuffer; }
    void setCommandEncoder(CommandEncoder&) const;
    bool isDestroyed() const;
private:
    QuerySet(id<MTLBuffer>, uint32_t, WGPUQueryType, Device&);
    QuerySet(id<MTLCounterSampleBuffer>, uint32_t, WGPUQueryType, Device&);
    QuerySet(Device&);

    const Ref<Device> m_device;
    id<MTLBuffer> m_visibilityBuffer { nil };
    id<MTLCounterSampleBuffer> m_timestampBuffer { nil };
    uint32_t m_count { 0 };
    const WGPUQueryType m_type { WGPUQueryType_Force32 };

    // rdar://91371495 is about how we can't just naively transform PassDescriptor.timestampWrites into MTLComputePassDescriptor.sampleBufferAttachments.
    // Instead, we can resolve all the information to a dummy counter sample buffer, and then internally remember that the data
    // is in a different place than where it's supposed to be. That's what this "overrides" vector is: A way to remember, when we resolve the data, that we
    // should resolve it from our dummy buffer instead of from where it's supposed to be.
    //
    // When rdar://91371495 is fixed, we can delete this indirection, and put the data directly where it's supposed to go.
    struct OverrideLocation {
        Ref<QuerySet> other;
        uint32_t otherIndex;
    };
    mutable WeakHashSet<CommandEncoder> m_commandEncoders;
    bool m_destroyed { false };
} SWIFT_SHARED_REFERENCE(refQuerySet, derefQuerySet);

} // namespace WebGPU

inline void refQuerySet(WebGPU::QuerySet* obj)
{
    WTF::ref(obj);
}

inline void derefQuerySet(WebGPU::QuerySet* obj)
{
    WTF::deref(obj);
}
