/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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
#include "WebGPUXRBindingImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUConvertToBackingContext.h"
#include "WebGPUDevice.h"
#include "WebGPUTextureFormat.h"
#include "WebGPUXRProjectionLayerImpl.h"
#include "WebGPUXRSubImageImpl.h"

namespace WebCore::WebGPU {

XRBindingImpl::XRBindingImpl(WebGPUPtr<WGPUXRBinding>&& binding, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(binding))
    , m_convertToBackingContext(convertToBackingContext)
{
}

XRBindingImpl::~XRBindingImpl() = default;

RefPtr<XRProjectionLayer> XRBindingImpl::createProjectionLayer(const XRProjectionLayerInit& init)
{
    Ref convertToBackingContext = m_convertToBackingContext;

    WGPUTextureFormat colorFormat = convertToBackingContext->convertToBacking(init.colorFormat);
    WGPUTextureFormat optionalDepthStencilFormat;
    if (init.depthStencilFormat)
        optionalDepthStencilFormat = convertToBackingContext->convertToBacking(*init.depthStencilFormat);
    WGPUTextureUsageFlags flags = convertToBackingContext->convertTextureUsageFlagsToBacking(init.textureUsage);
    return XRProjectionLayerImpl::create(adoptWebGPU(wgpuBindingCreateXRProjectionLayer(m_backing.get(), colorFormat, init.depthStencilFormat ? &optionalDepthStencilFormat : nullptr, flags, init.scaleFactor)), convertToBackingContext);
}

RefPtr<XRSubImage> XRBindingImpl::getSubImage(XRProjectionLayer&, WebCore::WebXRFrame&, std::optional<XREye>/* = "none"*/)
{
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

RefPtr<XRSubImage> XRBindingImpl::getViewSubImage(XRProjectionLayer& projectionLayer)
{
    auto& projectionLayerImpl = static_cast<XRProjectionLayerImpl&>(projectionLayer);
    return XRSubImageImpl::create(adoptWebGPU(wgpuBindingGetViewSubImage(m_backing.get(), projectionLayerImpl.backing())), Ref { m_convertToBackingContext });
}

TextureFormat XRBindingImpl::getPreferredColorFormat()
{
    return TextureFormat::Bgra8unormSRGB;
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
