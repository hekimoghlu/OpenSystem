/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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

#include "GPUShaderModule.h"
#include "WebGPUProgrammableStage.h"
#include <wtf/KeyValuePair.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

using GPUPipelineConstantValue = double; // May represent WGSLâ€™s bool, f32, i32, u32.

struct GPUProgrammableStage {
    WebGPU::ProgrammableStage convertToBacking() const
    {
        ASSERT(module);
        return {
            module->backing(),
            entryPoint,
            constants,
        };
    }

    WeakPtr<GPUShaderModule> module;
    std::optional<String> entryPoint;
    Vector<KeyValuePair<String, GPUPipelineConstantValue>> constants;
};

}
