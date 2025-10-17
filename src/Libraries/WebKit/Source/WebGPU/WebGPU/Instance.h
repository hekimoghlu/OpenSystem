/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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

#import <WebGPU/WebGPU.h>
#import <WebGPU/WebGPUExt.h>
#import <wtf/CompletionHandler.h>
#import <wtf/Deque.h>
#import <wtf/FastMalloc.h>
#import <wtf/Lock.h>
#import <wtf/MachSendRight.h>
#import <wtf/Ref.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/ThreadSafeRefCounted.h>
#import <wtf/WeakObjCPtr.h>
#import <wtf/WeakPtr.h>

struct WGPUInstanceImpl {
};

namespace WTF {
class MachSendRight;
}

namespace WebGPU {

class Adapter;
class Device;
class PresentationContext;

// https://gpuweb.github.io/gpuweb/#gpu
class Instance : public WGPUInstanceImpl, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<Instance> {
    WTF_MAKE_TZONE_ALLOCATED(Instance);
public:
    static Ref<Instance> create(const WGPUInstanceDescriptor&);
    static Ref<Instance> createInvalid()
    {
        return adoptRef(*new Instance());
    }

    virtual ~Instance();

    Ref<PresentationContext> createSurface(const WGPUSurfaceDescriptor&);
    void processEvents();
    void requestAdapter(const WGPURequestAdapterOptions&, CompletionHandler<void(WGPURequestAdapterStatus, Ref<Adapter>&&, String&&)>&& callback);

    bool isValid() const { return m_isValid; }
    void retainDevice(Device&, id<MTLCommandBuffer>);

    // This can be called on a background thread.
    using WorkItem = Function<void()>;
    void scheduleWork(WorkItem&&);
    const std::optional<const MachSendRight>& webProcessID() const;

private:
    Instance(WGPUScheduleWorkBlock, const WTF::MachSendRight* webProcessResourceOwner);
    explicit Instance();

    // This can be called on a background thread.
    void defaultScheduleWork(WGPUWorkItem&&);

    // This can be used on a background thread.
    Deque<WGPUWorkItem> m_pendingWork WTF_GUARDED_BY_LOCK(m_lock);
    using CommandBufferContainer = Vector<WeakObjCPtr<id<MTLCommandBuffer>>>;
    HashMap<RefPtr<Device>, CommandBufferContainer> retainedDeviceInstances WTF_GUARDED_BY_LOCK(m_lock);
    const std::optional<const MachSendRight> m_webProcessID;
    const WGPUScheduleWorkBlock m_scheduleWorkBlock;
    Lock m_lock;
    bool m_isValid { true };
};

} // namespace WebGPU
