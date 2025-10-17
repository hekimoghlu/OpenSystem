/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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

#include "GPUAutoLayoutMode.h"
#include "GPUObjectDescriptorBase.h"
#include "GPUPipelineLayout.h"
#include "WebGPUPipelineDescriptorBase.h"

#include <variant>

namespace WebCore {

using GPULayoutMode = std::variant<
    RefPtr<GPUPipelineLayout>,
    GPUAutoLayoutMode
>;

static WebGPU::PipelineLayout& convertPipelineLayoutToBacking(const GPULayoutMode& layout, const Ref<GPUPipelineLayout>& autoLayout)
{
    return *WTF::switchOn(layout, [](auto pipelineLayout) {
        return &pipelineLayout->backing();
    }, [&autoLayout](GPUAutoLayoutMode) {
        return &autoLayout->backing();
    });
}

struct GPUPipelineDescriptorBase : public GPUObjectDescriptorBase {
    WebGPU::PipelineDescriptorBase convertToBacking(const Ref<GPUPipelineLayout>& autoLayout) const
    {
        return {
            { label },
            &convertPipelineLayoutToBacking(layout, autoLayout)
        };
    }

    GPULayoutMode layout { nullptr };
};

}
