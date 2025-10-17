/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
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
#import "XRView.h"

#import "APIConversions.h"
#import "Device.h"
#import <wtf/CheckedArithmetic.h>
#import <wtf/StdLibExtras.h>

namespace WebGPU {

XRView::XRView(bool, Device& device)
    : m_device(device)
{
}

XRView::XRView(Device& device)
    : m_device(device)
{
}

XRView::~XRView() = default;

Ref<XRView> Device::createXRView()
{
    if (!isValid())
        return XRView::createInvalid(*this);

    return XRView::create(*this);
}

void XRView::setLabel(String&&)
{
}

bool XRView::isValid() const
{
    return true;
}

Device& XRView::device()
{
    return m_device;
}

} // namespace WebGPU

#pragma mark WGPU Stubs

void wgpuXRViewReference(WGPUXRView binding)
{
    WebGPU::fromAPI(binding).ref();
}

void wgpuXRViewRelease(WGPUXRView binding)
{
    WebGPU::fromAPI(binding).deref();
}
