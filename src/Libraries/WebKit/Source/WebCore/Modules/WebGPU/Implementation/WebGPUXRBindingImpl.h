/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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

#include "WebGPUPtr.h"
#include "WebGPUXRBinding.h"

#include <WebGPU/WebGPU.h>

#if PLATFORM(COCOA)
#include <wtf/MachSendRight.h>
#include <wtf/RetainPtr.h>
#include <wtf/spi/cocoa/IOSurfaceSPI.h>
#endif

namespace WebCore {
class Device;
class NativeImage;
}

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class XRBindingImpl final : public XRBinding {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static Ref<XRBindingImpl> create(WebGPUPtr<WGPUXRBinding>&& binding, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new XRBindingImpl(WTFMove(binding), convertToBackingContext));
    }

    virtual ~XRBindingImpl();

private:
    friend class DowncastConvertToBackingContext;

    explicit XRBindingImpl(WebGPUPtr<WGPUXRBinding>&&, ConvertToBackingContext&);

    XRBindingImpl(const XRBindingImpl&) = delete;
    XRBindingImpl(XRBindingImpl&&) = delete;
    XRBindingImpl& operator=(const XRBindingImpl&) = delete;
    XRBindingImpl& operator=(XRBindingImpl&&) = delete;

    WGPUXRBinding backing() const { return m_backing.get(); }

    RefPtr<XRProjectionLayer> createProjectionLayer(const XRProjectionLayerInit&) final;
    RefPtr<XRSubImage> getSubImage(XRProjectionLayer&, WebCore::WebXRFrame&, std::optional<XREye>/* = "none"*/) final;
    RefPtr<XRSubImage> getViewSubImage(XRProjectionLayer&) final;
    TextureFormat getPreferredColorFormat() final;

    WebGPUPtr<WGPUXRBinding> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
