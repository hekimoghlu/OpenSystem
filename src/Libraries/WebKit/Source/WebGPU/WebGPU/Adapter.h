/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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

#import "HardwareCapabilities.h"
#import <Metal/Metal.h>
#import <wtf/CompletionHandler.h>
#import <wtf/FastMalloc.h>
#import <wtf/Ref.h>
#import <wtf/RefCounted.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakPtr.h>

struct WGPUAdapterImpl {
};

namespace WebGPU {

class Device;
class Instance;

// https://gpuweb.github.io/gpuweb/#gpuadapter
class Adapter : public WGPUAdapterImpl, public RefCounted<Adapter> {
    WTF_MAKE_TZONE_ALLOCATED(Adapter);
public:
    static Ref<Adapter> create(id<MTLDevice> device, Instance& instance, bool xrCompatible, HardwareCapabilities&& capabilities)
    {
        return adoptRef(*new Adapter(device, instance, xrCompatible, WTFMove(capabilities)));
    }
    static Ref<Adapter> createInvalid(Instance& instance)
    {
        return adoptRef(*new Adapter(instance));
    }

    ~Adapter();

    size_t enumerateFeatures(WGPUFeatureName* features);
    bool getLimits(WGPUSupportedLimits&);
    void getProperties(WGPUAdapterProperties&);
    bool hasFeature(WGPUFeatureName);
    void requestDevice(const WGPUDeviceDescriptor&, CompletionHandler<void(WGPURequestDeviceStatus, Ref<Device>&&, String&&)>&& callback);

    bool isValid() const { return m_device; }
    void makeInvalid() { m_device = nil; }
    bool isXRCompatible() const;

    RefPtr<Instance> instance() const { return m_instance.get(); }
    ThreadSafeWeakPtr<Instance> weakInstance() const { return m_instance; }

private:
    Adapter(id<MTLDevice>, Instance&, bool xrCompatible, HardwareCapabilities&&);
    Adapter(Instance&);

    id<MTLDevice> m_device { nil };
    const ThreadSafeWeakPtr<Instance> m_instance;

    const HardwareCapabilities m_capabilities { };
    bool m_deviceRequested { false };
    bool m_xrCompatible { false };
};

} // namespace WebGPU
