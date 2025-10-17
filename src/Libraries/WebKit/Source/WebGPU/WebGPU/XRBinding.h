/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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

#import <utility>
#import <wtf/CompletionHandler.h>
#import <wtf/FastMalloc.h>
#import <wtf/Ref.h>
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/WeakPtr.h>

struct WGPUXRBindingImpl {
};

namespace WebGPU {

class CommandEncoder;
class Device;
enum class XREye : uint8_t;
class XRProjectionLayer;
class XRSubImage;

class XRBinding : public RefCountedAndCanMakeWeakPtr<XRBinding>, public WGPUXRBindingImpl {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static Ref<XRBinding> create(Device& device)
    {
        return adoptRef(*new XRBinding(true, device));
    }
    static Ref<XRBinding> createInvalid(Device& device)
    {
        return adoptRef(*new XRBinding(device));
    }

    ~XRBinding();

    void setLabel(String&&);

    bool isValid() const;
    Ref<XRProjectionLayer> createXRProjectionLayer(WGPUTextureFormat, WGPUTextureFormat*, WGPUTextureUsageFlags, double);
    RefPtr<XRSubImage> getViewSubImage(XRProjectionLayer&);
    Device& device() { return m_device; }
    Ref<Device> protectedDevice() { return m_device; }


private:
    XRBinding(bool, Device&);
    XRBinding(Device&);

    Ref<Device> m_device;
};

} // namespace WebGPU
