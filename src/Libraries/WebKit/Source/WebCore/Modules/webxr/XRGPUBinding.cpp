/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
#include "config.h"
#include "XRGPUBinding.h"

#if ENABLE(WEBXR_LAYERS)

#include "GPUDevice.h"
#include "WebGPUXRBinding.h"
#include "WebGPUXREye.h"
#include "WebGPUXRView.h"
#include "WebXRFrame.h"
#include "WebXRView.h"
#include "XRCompositionLayer.h"
#include "XRGPUProjectionLayerInit.h"
#include "XRGPUSubImage.h"
#include "XRProjectionLayer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static WebGPU::XREye convertToBacking(XREye eye)
{
    switch (eye) {
    case PlatformXR::Eye::None:
        return WebGPU::XREye::None;
    case PlatformXR::Eye::Left:
        return WebGPU::XREye::Left;
    case PlatformXR::Eye::Right:
        return WebGPU::XREye::Right;
    }
}

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(XRGPUBinding);

XRGPUBinding::XRGPUBinding(const WebXRSession& session, GPUDevice& device)
    : m_backing(device.createXRBinding(session))
    , m_session(&session)
    , m_device(device)
{
}

GPUDevice& XRGPUBinding::device()
{
    return m_device;
}

ExceptionOr<Ref<XRProjectionLayer>> XRGPUBinding::createProjectionLayer(ScriptExecutionContext& scriptExecutionContext, std::optional<XRGPUProjectionLayerInit> init)
{
    if (!m_backing)
        return Exception { ExceptionCode::AbortError };

    WebGPU::XRProjectionLayerInit convertedInit;
    if (init)
        convertedInit = init->convertToBacking();
    RefPtr projectionLayer = m_backing->createProjectionLayer(convertedInit);
    if (!projectionLayer)
        return Exception { ExceptionCode::AbortError };

    m_init = init;
    return XRProjectionLayer::create(scriptExecutionContext, projectionLayer.releaseNonNull());
}

double XRGPUBinding::nativeProjectionScaleFactor() const
{
    return m_init ? m_init->scaleFactor : 1.0;
}

RefPtr<XRGPUSubImage> XRGPUBinding::getSubImage(XRCompositionLayer&, WebXRFrame&, std::optional<XREye>/* = "none"*/)
{
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

ExceptionOr<Ref<XRGPUSubImage>> XRGPUBinding::getViewSubImage(XRProjectionLayer& projectionLayer, WebXRView& xrView)
{
    if (!m_backing)
        return Exception { ExceptionCode::AbortError };

    RefPtr subImage = m_backing->getViewSubImage(projectionLayer.backing());
    return XRGPUSubImage::create(subImage.releaseNonNull(), convertToBacking(xrView.eye()), m_device);
}

GPUTextureFormat XRGPUBinding::getPreferredColorFormat()
{
    return GPUTextureFormat::Bgra8unormSRGB;
}

} // namespace WebCore

#endif // ENABLE(WEBXR_LAYERS)

