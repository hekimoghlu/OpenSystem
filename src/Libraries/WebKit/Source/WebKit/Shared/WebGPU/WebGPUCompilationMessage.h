/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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

#if ENABLE(GPU_PROCESS)

#include <WebCore/WebGPUCompilationMessageType.h>
#include <cstdint>
#include <optional>
#include <wtf/text/WTFString.h>

namespace WebKit::WebGPU {

struct CompilationMessage {
    String message;
    WebCore::WebGPU::CompilationMessageType type { WebCore::WebGPU::CompilationMessageType::Error };
    uint64_t lineNum { 0 };
    uint64_t linePos { 0 };
    uint64_t offset { 0 };
    uint64_t length { 0 };
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
