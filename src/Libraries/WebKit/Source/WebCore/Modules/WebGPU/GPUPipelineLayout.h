/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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

#include "WebGPUPipelineLayout.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUPipelineLayout : public RefCounted<GPUPipelineLayout> {
public:
    static Ref<GPUPipelineLayout> create(Ref<WebGPU::PipelineLayout>&& backing)
    {
        return adoptRef(*new GPUPipelineLayout(WTFMove(backing)));
    }

    String label() const;
    void setLabel(String&&);

    WebGPU::PipelineLayout& backing() { return m_backing; }
    const WebGPU::PipelineLayout& backing() const { return m_backing; }

private:
    GPUPipelineLayout(Ref<WebGPU::PipelineLayout>&& backing)
        : m_backing(WTFMove(backing))
    {
    }

    Ref<WebGPU::PipelineLayout> m_backing;
};

}
