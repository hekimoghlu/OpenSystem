/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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

#include "GPUCompilationMessageType.h"
#include "WebGPUCompilationMessage.h"
#include <cstdint>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUCompilationMessage : public RefCounted<GPUCompilationMessage> {
public:
    static Ref<GPUCompilationMessage> create(WebGPU::CompilationMessage& backing)
    {
        return adoptRef(*new GPUCompilationMessage(backing));
    }

    const String& message() const;
    GPUCompilationMessageType type() const;
    uint64_t lineNum() const;
    uint64_t linePos() const;
    uint64_t offset() const;
    uint64_t length() const;

    WebGPU::CompilationMessage& backing() { return m_backing; }
    const WebGPU::CompilationMessage& backing() const { return m_backing; }

private:
    GPUCompilationMessage(WebGPU::CompilationMessage& backing)
        : m_backing(backing)
    {
    }

    Ref<WebGPU::CompilationMessage> m_backing;
};

}
