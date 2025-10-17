/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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

#include "GPUBindGroupLayout.h"
#include "WebGPURenderPipeline.h"
#include <cstdint>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPURenderPipeline : public RefCounted<GPURenderPipeline> {
public:
    static Ref<GPURenderPipeline> create(Ref<WebGPU::RenderPipeline>&& backing)
    {
        return adoptRef(*new GPURenderPipeline(WTFMove(backing)));
    }

    String label() const;
    void setLabel(String&&);

    Ref<GPUBindGroupLayout> getBindGroupLayout(uint32_t index);

    WebGPU::RenderPipeline& backing() { return m_backing; }
    const WebGPU::RenderPipeline& backing() const { return m_backing; }

private:
    GPURenderPipeline(Ref<WebGPU::RenderPipeline>&& backing)
        : m_backing(WTFMove(backing))
    {
    }

    Ref<WebGPU::RenderPipeline> m_backing;
};

}
