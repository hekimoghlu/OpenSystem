/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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

#if ENABLE(WEBXR_LAYERS)

#include "GPUTexture.h"
#include "GPUTextureViewDescriptor.h"
#include "WebGPUXREye.h"
#include "WebGPUXRSubImage.h"
#include "XRSubImage.h"

#include <wtf/TZoneMalloc.h>

namespace WebCore {

class GPUTexture;

// https://github.com/immersive-web/WebXR-WebGPU-Binding/blob/main/explainer.md
class XRGPUSubImage : public XRSubImage {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XRGPUSubImage);
public:
    static Ref<XRGPUSubImage> create(Ref<WebGPU::XRSubImage>&& backing, WebGPU::XREye eye, GPUDevice& device)
    {
        return adoptRef(*new XRGPUSubImage(WTFMove(backing), eye, device));
    }

    const WebXRViewport& viewport() const final;
    ExceptionOr<Ref<GPUTexture>> colorTexture();
    RefPtr<GPUTexture> depthStencilTexture();
    RefPtr<GPUTexture> motionVectorTexture();

    const GPUTextureViewDescriptor& getViewDescriptor() const;
private:
    XRGPUSubImage(Ref<WebGPU::XRSubImage>&&, WebGPU::XREye, GPUDevice&);

    Ref<WebGPU::XRSubImage> m_backing;
    Ref<GPUDevice> m_device;
    const GPUTextureViewDescriptor m_descriptor;
    Ref<WebXRViewport> m_viewport;
};

} // namespace WebCore

#endif // ENABLE(WEBXR_LAYERS)
