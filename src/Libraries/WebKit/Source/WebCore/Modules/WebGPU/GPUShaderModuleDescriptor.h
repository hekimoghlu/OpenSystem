/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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

#include "GPUObjectDescriptorBase.h"
#include "GPUShaderModuleCompilationHint.h"
#include "WebGPUShaderModuleDescriptor.h"
#include <JavaScriptCore/JSObject.h>
#include <JavaScriptCore/Strong.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct GPUShaderModuleDescriptor : public GPUObjectDescriptorBase {
    WebGPU::ShaderModuleDescriptor convertToBacking(const Ref<GPUPipelineLayout>& autoLayout) const
    {
        return {
            { label },
            code,
            // FIXME: Handle the sourceMap.
            hints.map([&autoLayout](auto& hint) {
                return KeyValuePair<String, WebGPU::ShaderModuleCompilationHint>(hint.key, hint.value.convertToBacking(autoLayout));
            }),
        };
    }

    String code;
    JSC::Strong<JSC::JSObject> sourceMap;
    Vector<KeyValuePair<String, GPUShaderModuleCompilationHint>> hints;
};

}
