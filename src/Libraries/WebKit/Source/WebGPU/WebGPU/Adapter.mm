/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#import "Adapter.h"

#import "APIConversions.h"
#import "Device.h"
#import "Instance.h"
#import <algorithm>
#import <wtf/StdLibExtras.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Adapter);

Adapter::Adapter(id<MTLDevice> device, Instance& instance, bool xrCompatible, HardwareCapabilities&& capabilities)
    : m_device(device)
    , m_instance(&instance)
    , m_capabilities(WTFMove(capabilities))
    , m_xrCompatible(xrCompatible)
{
}

Adapter::Adapter(Instance& instance)
    : m_instance(&instance)
{
}

Adapter::~Adapter() = default;

size_t Adapter::enumerateFeatures(WGPUFeatureName* features)
{
    // The API contract for this requires that sufficient space has already been allocated for the output.
    // This requires the caller calling us twice: once to get the amount of space to allocate, and once to fill the space.
    if (features)
        std::copy(m_capabilities.features.begin(), m_capabilities.features.end(), features);
    return m_capabilities.features.size();
}

bool Adapter::getLimits(WGPUSupportedLimits& limits)
{
    if (limits.nextInChain != nullptr)
        return false;

    limits.limits = m_capabilities.limits;
    return true;
}

void Adapter::getProperties(WGPUAdapterProperties& properties)
{
    // FIXME: What should the vendorID and deviceID be?
    properties.vendorID = 0;
    properties.deviceID = 0;
    properties.name = m_device.name.UTF8String;
    properties.driverDescription = "";
    properties.adapterType = m_device.hasUnifiedMemory ? WGPUAdapterType_IntegratedGPU : WGPUAdapterType_DiscreteGPU;
    properties.backendType = WGPUBackendType_Metal;
}

bool Adapter::hasFeature(WGPUFeatureName feature)
{
    return m_capabilities.features.contains(feature);
}

void Adapter::requestDevice(const WGPUDeviceDescriptor& descriptor, CompletionHandler<void(WGPURequestDeviceStatus, Ref<Device>&&, String&&)>&& callback)
{
    if (descriptor.nextInChain) {
        callback(WGPURequestDeviceStatus_Error, Device::createInvalid(*this), "Unknown descriptor type"_s);
        return;
    }

    if (m_deviceRequested) {
        callback(WGPURequestDeviceStatus_Error, Device::createInvalid(*this), "Adapter can only request one device"_s);
        makeInvalid();
        return;
    }

    WGPULimits limits { };

    if (descriptor.requiredLimits) {
        if (descriptor.requiredLimits->nextInChain) {
            callback(WGPURequestDeviceStatus_Error, Device::createInvalid(*this), "Unknown descriptor type"_s);
            return;
        }

        if (!WebGPU::isValid(descriptor.requiredLimits->limits)) {
            callback(WGPURequestDeviceStatus_Error, Device::createInvalid(*this), "Device does not support requested limits"_s);
            return;
        }

        if (anyLimitIsBetterThan(descriptor.requiredLimits->limits, m_capabilities.limits)) {
            callback(WGPURequestDeviceStatus_Error, Device::createInvalid(*this), "Device does not support requested limits"_s);
            return;
        }

        limits = descriptor.requiredLimits->limits;
    } else
        limits = defaultLimits();

    Vector<WGPUFeatureName> features(descriptor.requiredFeaturesSpan());
    if (includesUnsupportedFeatures(features, m_capabilities.features)) {
        callback(WGPURequestDeviceStatus_Error, Device::createInvalid(*this), "Device does not support requested features"_s);
        return;
    }

    HardwareCapabilities capabilities {
        limits,
        WTFMove(features),
        m_capabilities.baseCapabilities,
    };

    auto label = fromAPI(descriptor.label);
    m_deviceRequested = true;
    // FIXME: this should be asynchronous - https://bugs.webkit.org/show_bug.cgi?id=233621
    callback(WGPURequestDeviceStatus_Success, Device::create(this->m_device, WTFMove(label), WTFMove(capabilities), *this), { });
}

bool Adapter::isXRCompatible() const
{
    return m_xrCompatible;
}

} // namespace WebGPU

#pragma mark WGPU Stubs

void wgpuAdapterReference(WGPUAdapter adapter)
{
    WebGPU::fromAPI(adapter).ref();
}

void wgpuAdapterRelease(WGPUAdapter adapter)
{
    WebGPU::fromAPI(adapter).deref();
}

size_t wgpuAdapterEnumerateFeatures(WGPUAdapter adapter, WGPUFeatureName* features)
{
    return WebGPU::protectedFromAPI(adapter)->enumerateFeatures(features);
}

WGPUBool wgpuAdapterGetLimits(WGPUAdapter adapter, WGPUSupportedLimits* limits)
{
    return WebGPU::protectedFromAPI(adapter)->getLimits(*limits);
}

void wgpuAdapterGetProperties(WGPUAdapter adapter, WGPUAdapterProperties* properties)
{
    WebGPU::protectedFromAPI(adapter)->getProperties(*properties);
}

WGPUBool wgpuAdapterHasFeature(WGPUAdapter adapter, WGPUFeatureName feature)
{
    return WebGPU::protectedFromAPI(adapter)->hasFeature(feature);
}

void wgpuAdapterRequestDevice(WGPUAdapter adapter, const WGPUDeviceDescriptor* descriptor, WGPURequestDeviceCallback callback, void* userdata)
{
    WebGPU::protectedFromAPI(adapter)->requestDevice(*descriptor, [callback, userdata](WGPURequestDeviceStatus status, Ref<WebGPU::Device>&& device, String&& message) {
        callback(status, WebGPU::releaseToAPI(WTFMove(device)), message.utf8().data(), userdata);
    });
}

void wgpuAdapterRequestDeviceWithBlock(WGPUAdapter adapter, WGPUDeviceDescriptor const * descriptor, WGPURequestDeviceBlockCallback callback)
{
    WebGPU::protectedFromAPI(adapter)->requestDevice(*descriptor, [callback = WebGPU::fromAPI(WTFMove(callback))](WGPURequestDeviceStatus status, Ref<WebGPU::Device>&& device, String&& message) {
        callback(status, WebGPU::releaseToAPI(WTFMove(device)), message.utf8().data());
    });
}

WGPUBool wgpuAdapterXRCompatible(WGPUAdapter adapter)
{
    return WebGPU::protectedFromAPI(adapter)->isXRCompatible();
}
