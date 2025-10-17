/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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

#include "WebGPUCompositorIntegration.h"
#include <optional>
#include <wtf/MachSendRight.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
namespace WebGPU {
class Device;

enum class TextureFormat : uint8_t;
}

class DestinationColorSpace;
class ImageBuffer;

class GPUCompositorIntegration : public RefCounted<GPUCompositorIntegration> {
public:
    static Ref<GPUCompositorIntegration> create(Ref<WebGPU::CompositorIntegration>&& backing)
    {
        return adoptRef(*new GPUCompositorIntegration(WTFMove(backing)));
    }

#if PLATFORM(COCOA)
    Vector<MachSendRight> recreateRenderBuffers(int width, int height, WebCore::DestinationColorSpace&&, WebCore::AlphaPremultiplication, WebCore::WebGPU::TextureFormat, WebCore::WebGPU::Device&) const;
#endif

    void prepareForDisplay(uint32_t frameIndex, CompletionHandler<void()>&&);

    WebGPU::CompositorIntegration& backing() { return m_backing; }
    const WebGPU::CompositorIntegration& backing() const { return m_backing; }

    void paintCompositedResultsToCanvas(WebCore::ImageBuffer&, uint32_t);

private:
    GPUCompositorIntegration(Ref<WebGPU::CompositorIntegration>&& backing)
        : m_backing(WTFMove(backing))
    {
    }

    Ref<WebGPU::CompositorIntegration> m_backing;
};

}
