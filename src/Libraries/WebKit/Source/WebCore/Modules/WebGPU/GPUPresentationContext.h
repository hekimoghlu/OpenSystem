/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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

#include "GPUTexture.h"
#include "GPUTextureDescriptor.h"
#include "WebGPUPresentationContext.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

#if PLATFORM(COCOA)
#include <wtf/MachSendRight.h>
#endif

namespace WebCore {

struct GPUCanvasConfiguration;
class GPUDevice;
class GPUTexture;

class GPUPresentationContext : public RefCounted<GPUPresentationContext> {
public:
    static Ref<GPUPresentationContext> create(Ref<WebGPU::PresentationContext>&& backing)
    {
        return adoptRef(*new GPUPresentationContext(WTFMove(backing)));
    }

    WARN_UNUSED_RETURN bool configure(const GPUCanvasConfiguration&, GPUIntegerCoordinate, GPUIntegerCoordinate, bool);
    void unconfigure();

    RefPtr<GPUTexture> getCurrentTexture(uint32_t);
    void present(uint32_t frameIndex, bool presentBacking = false);

    WebGPU::PresentationContext& backing() { return m_backing; }
    const WebGPU::PresentationContext& backing() const { return m_backing; }

private:
    GPUPresentationContext(Ref<WebGPU::PresentationContext>&& backing)
        : m_backing(WTFMove(backing))
    {
    }

    Ref<WebGPU::PresentationContext> m_backing;
    RefPtr<GPUTexture> m_currentTexture;
    RefPtr<const GPUDevice> m_device;
    GPUTextureDescriptor m_textureDescriptor;
};

}
