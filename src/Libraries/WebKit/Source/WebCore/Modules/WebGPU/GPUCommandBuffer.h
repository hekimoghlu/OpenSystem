/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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

#include "GPUCommandEncoder.h"
#include "WebGPUCommandBuffer.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

namespace WebGPU {
class CommandEncoder;
}

class GPUCommandBuffer : public RefCounted<GPUCommandBuffer> {
public:
    static Ref<GPUCommandBuffer> create(Ref<WebGPU::CommandBuffer>&& backing, GPUCommandEncoder& encoder)
    {
        return adoptRef(*new GPUCommandBuffer(WTFMove(backing), encoder));
    }

    String label() const;
    void setLabel(String&&);

    WebGPU::CommandBuffer& backing() { return m_backing; }
    const WebGPU::CommandBuffer& backing() const { return m_backing; }
    void setBacking(WebGPU::CommandEncoder&, WebGPU::CommandBuffer&);

private:
    GPUCommandBuffer(Ref<WebGPU::CommandBuffer>&& backing, GPUCommandEncoder& encoder)
        : m_backing(WTFMove(backing))
        , m_encoder(encoder)
    {
    }

    Ref<WebGPU::CommandBuffer> m_backing;
    Ref<GPUCommandEncoder> m_encoder;
};

}
