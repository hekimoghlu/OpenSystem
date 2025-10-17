/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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

#include "GPUDepthStencilState.h"
#include "GPUFragmentState.h"
#include "GPUMultisampleState.h"
#include "GPUPipelineDescriptorBase.h"
#include "GPUPrimitiveState.h"
#include "GPUVertexState.h"
#include "WebGPURenderPipelineDescriptor.h"
#include <optional>

namespace WebCore {

struct GPURenderPipelineDescriptor : public GPUPipelineDescriptorBase {
    WebGPU::RenderPipelineDescriptor convertToBacking(const Ref<GPUPipelineLayout>& autoLayout) const
    {
        return {
            {
                { label },
                &convertPipelineLayoutToBacking(layout, autoLayout),
            },
            vertex.convertToBacking(),
            primitive ? std::optional { primitive->convertToBacking() } : std::nullopt,
            depthStencil ? std::optional { depthStencil->convertToBacking() } : std::nullopt,
            multisample ? std::optional { multisample->convertToBacking() } : std::nullopt,
            fragment ? std::optional { fragment->convertToBacking() } : std::nullopt,
        };
    }

    GPUVertexState vertex;
    std::optional<GPUPrimitiveState> primitive;
    std::optional<GPUDepthStencilState> depthStencil;
    std::optional<GPUMultisampleState> multisample;
    std::optional<GPUFragmentState> fragment;
};

}
