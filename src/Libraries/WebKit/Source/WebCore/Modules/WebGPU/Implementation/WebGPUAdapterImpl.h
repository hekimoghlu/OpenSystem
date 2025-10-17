/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUAdapter.h"
#include "WebGPUPtr.h"
#include <WebGPU/WebGPU.h>
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class AdapterImpl final : public Adapter {
    WTF_MAKE_TZONE_ALLOCATED(AdapterImpl);
public:
    static Ref<AdapterImpl> create(WebGPUPtr<WGPUAdapter>&& adapter, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new AdapterImpl(WTFMove(adapter), convertToBackingContext));
    }

    virtual ~AdapterImpl();

private:
    friend class DowncastConvertToBackingContext;

    AdapterImpl(WebGPUPtr<WGPUAdapter>&&, ConvertToBackingContext&);

    AdapterImpl(const AdapterImpl&) = delete;
    AdapterImpl(AdapterImpl&&) = delete;
    AdapterImpl& operator=(const AdapterImpl&) = delete;
    AdapterImpl& operator=(AdapterImpl&&) = delete;

    WGPUAdapter backing() const { return m_backing.get(); }
    bool xrCompatible() final;

    void requestDevice(const DeviceDescriptor&, CompletionHandler<void(RefPtr<Device>&&)>&&) final;

    WebGPUPtr<WGPUAdapter> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
