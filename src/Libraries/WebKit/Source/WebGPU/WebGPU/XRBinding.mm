/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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
#import "XRBinding.h"

#import "APIConversions.h"
#import "Device.h"
#import <wtf/CheckedArithmetic.h>
#import <wtf/StdLibExtras.h>

namespace WebGPU {

XRBinding::XRBinding(bool, Device& device)
    : m_device(device)
{
}

XRBinding::XRBinding(Device& device)
    : m_device(device)
{
}

XRBinding::~XRBinding() = default;

Ref<XRBinding> Device::createXRBinding()
{
    if (!isValid())
        return XRBinding::createInvalid(*this);

    return XRBinding::create(*this);
}

void XRBinding::setLabel(String&&)
{
}

bool XRBinding::isValid() const
{
    return true;
}

} // namespace WebGPU

#pragma mark WGPU Stubs

void wgpuXRBindingReference(WGPUXRBinding binding)
{
    WebGPU::fromAPI(binding).ref();
}

void wgpuXRBindingRelease(WGPUXRBinding binding)
{
    WebGPU::fromAPI(binding).deref();
}

WGPUXRProjectionLayer wgpuBindingCreateXRProjectionLayer(WGPUXRBinding binding, WGPUTextureFormat colorFormat, WGPUTextureFormat* optionalDepthStencilFormat, WGPUTextureUsageFlags flags, double scale)
{
    return WebGPU::releaseToAPI(WebGPU::protectedFromAPI(binding)->createXRProjectionLayer(colorFormat, optionalDepthStencilFormat, flags, scale));
}

WGPUXRSubImage wgpuBindingGetViewSubImage(WGPUXRBinding binding, WGPUXRProjectionLayer layer)
{
    return WebGPU::releaseToAPI(WebGPU::protectedFromAPI(binding)->getViewSubImage(WebGPU::protectedFromAPI(layer)));
}

