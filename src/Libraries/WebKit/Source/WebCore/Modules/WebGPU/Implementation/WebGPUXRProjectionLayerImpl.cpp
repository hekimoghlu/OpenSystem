/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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
#include "WebGPUXRProjectionLayerImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUConvertToBackingContext.h"
#include "WebGPUDevice.h"
#include "WebGPUTextureFormat.h"
#include "WebXRRigidTransform.h"
#include <wtf/MachSendRight.h>

namespace WebCore::WebGPU {

XRProjectionLayerImpl::XRProjectionLayerImpl(WebGPUPtr<WGPUXRProjectionLayer>&& projectionLayer, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(projectionLayer))
    , m_convertToBackingContext(convertToBackingContext)
{
}

XRProjectionLayerImpl::~XRProjectionLayerImpl() = default;

uint32_t XRProjectionLayerImpl::textureWidth() const
{
    return 0;
}

uint32_t XRProjectionLayerImpl::textureHeight() const
{
    return 0;
}

uint32_t XRProjectionLayerImpl::textureArrayLength() const
{
#if PLATFORM(IOS_FAMILY_SIMULATOR)
    return 1;
#else
    return 2;
#endif
}

bool XRProjectionLayerImpl::ignoreDepthValues() const
{
    return false;
}

std::optional<float> XRProjectionLayerImpl::fixedFoveation() const
{
    return 0.f;
}

void XRProjectionLayerImpl::setFixedFoveation(std::optional<float>)
{
    return;
}

WebXRRigidTransform* XRProjectionLayerImpl::deltaPose() const
{
#if ENABLE(WEBXR)
    return m_webXRRigidTransform.get();
#else
    return nullptr;
#endif
}

void XRProjectionLayerImpl::setDeltaPose(WebXRRigidTransform* pose)
{
#if ENABLE(WEBXR)
    m_webXRRigidTransform = pose;
#else
    UNUSED_PARAM(pose);
#endif
}

// WebXRLayer
void XRProjectionLayerImpl::startFrame(size_t frameIndex, MachSendRight&& colorBuffer, MachSendRight&& depthBuffer, MachSendRight&& completionSyncEvent, size_t reusableTextureIndex)
{
    wgpuXRProjectionLayerStartFrame(m_backing.get(), frameIndex, WTFMove(colorBuffer), WTFMove(depthBuffer), WTFMove(completionSyncEvent), reusableTextureIndex);
}

void XRProjectionLayerImpl::endFrame()
{
    return;
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
