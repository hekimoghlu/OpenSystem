/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
#include "RemoteXRProjectionLayerProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteGPUProxy.h"
#include "RemoteXRProjectionLayerMessages.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/ImageBuffer.h>
#include <WebCore/PlatformXR.h>
#include <WebCore/WebGPUTextureFormat.h>
#include <wtf/MachSendRight.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteXRProjectionLayerProxy);

RemoteXRProjectionLayerProxy::RemoteXRProjectionLayerProxy(Ref<RemoteGPUProxy>&& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_parent(WTFMove(parent))
{
}

RemoteXRProjectionLayerProxy::~RemoteXRProjectionLayerProxy()
{
    auto sendResult = send(Messages::RemoteXRProjectionLayer::Destruct());
    UNUSED_VARIABLE(sendResult);
}

#if PLATFORM(COCOA)
void RemoteXRProjectionLayerProxy::startFrame(size_t frameIndex, MachSendRight&& colorBuffer, MachSendRight&& depthBuffer, MachSendRight&& completionSyncEvent, size_t reusableTextureIndex)
{
    auto sendResult = send(Messages::RemoteXRProjectionLayer::StartFrame(frameIndex, WTFMove(colorBuffer), WTFMove(depthBuffer), WTFMove(completionSyncEvent), reusableTextureIndex));
    UNUSED_VARIABLE(sendResult);
}
#endif

void RemoteXRProjectionLayerProxy::endFrame()
{
    auto sendResult = send(Messages::RemoteXRProjectionLayer::EndFrame());
    UNUSED_VARIABLE(sendResult);
}

uint32_t RemoteXRProjectionLayerProxy::textureWidth() const
{
    return 0;
}

uint32_t RemoteXRProjectionLayerProxy::textureHeight() const
{
    return 0;
}

uint32_t RemoteXRProjectionLayerProxy::textureArrayLength() const
{
    return 0;
}

bool RemoteXRProjectionLayerProxy::ignoreDepthValues() const
{
    return false;
}

std::optional<float> RemoteXRProjectionLayerProxy::fixedFoveation() const
{
    return 0.f;
}

void RemoteXRProjectionLayerProxy::setFixedFoveation(std::optional<float>)
{
    return;
}

WebCore::WebXRRigidTransform* RemoteXRProjectionLayerProxy::deltaPose() const
{
    return nullptr;
}

void RemoteXRProjectionLayerProxy::setDeltaPose(WebCore::WebXRRigidTransform*)
{
    return;
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
