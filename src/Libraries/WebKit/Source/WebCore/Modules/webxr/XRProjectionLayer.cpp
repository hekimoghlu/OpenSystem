/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
#include "XRProjectionLayer.h"

#if ENABLE(WEBXR_LAYERS)

#include "PlatformXR.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(XRProjectionLayer);

XRProjectionLayer::XRProjectionLayer(ScriptExecutionContext& scriptExecutionContext, Ref<WebCore::WebGPU::XRProjectionLayer>&& backing)
    : XRCompositionLayer(&scriptExecutionContext)
    , m_backing(WTFMove(backing))
{
}

XRProjectionLayer::~XRProjectionLayer() = default;

void XRProjectionLayer::startFrame(PlatformXR::FrameData& data)
{
    static constexpr auto defaultLayerHandle = 1;
    auto it = data.layers.find(defaultLayerHandle);
    if (it == data.layers.end()) {
        // For some reason the device didn't provide a texture for this frame.
        // The frame is ignored and the device can recover the texture in future frames;
        return;
    }

    auto& frameData = it->value;
    if (frameData->layerSetup && frameData->textureData) {
        auto& textureData = frameData->textureData;
        m_backing->startFrame(frameData->renderingFrameIndex, WTFMove(textureData->colorTexture.handle), WTFMove(textureData->depthStencilBuffer.handle), WTFMove(frameData->layerSetup->completionSyncEvent), textureData->reusableTextureIndex);
    }
}

PlatformXR::Device::Layer XRProjectionLayer::endFrame()
{
    m_backing->endFrame();
    return PlatformXR::Device::Layer {
        .handle = 0,
        .visible = true,
        .views = { },
    };
}

uint32_t XRProjectionLayer::textureWidth() const
{
    return m_backing->textureWidth();
}

uint32_t XRProjectionLayer::textureHeight() const
{
    return m_backing->textureHeight();
}

uint32_t XRProjectionLayer::textureArrayLength() const
{
#if PLATFORM(IOS_FAMILY_SIMULATOR)
    ASSERT(m_backing->textureArrayLength() == 1);
#else
    ASSERT(m_backing->textureArrayLength() == 2);
#endif
    return m_backing->textureArrayLength();
}

bool XRProjectionLayer::ignoreDepthValues() const
{
    RELEASE_ASSERT_NOT_REACHED();
}

std::optional<float> XRProjectionLayer::fixedFoveation() const
{
    RELEASE_ASSERT_NOT_REACHED();
}

void XRProjectionLayer::setFixedFoveation(std::optional<float>)
{
    RELEASE_ASSERT_NOT_REACHED();
}

WebXRRigidTransform* XRProjectionLayer::deltaPose() const
{
    RELEASE_ASSERT_NOT_REACHED();
}

void XRProjectionLayer::setDeltaPose(WebXRRigidTransform*)
{
    RELEASE_ASSERT_NOT_REACHED();
}

WebCore::WebGPU::XRProjectionLayer& XRProjectionLayer::backing()
{
    return m_backing;
}

} // namespace WebCore

#endif // ENABLE(WEBXR_LAYERS)
