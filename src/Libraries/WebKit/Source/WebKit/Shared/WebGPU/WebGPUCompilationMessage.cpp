/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
#include "config.h"
#include "WebGPUCompilationMessage.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUCompilationMessage.h>

namespace WebKit::WebGPU {

std::optional<CompilationMessage> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::CompilationMessage& compilationMessage)
{
    return { { compilationMessage.message(), compilationMessage.type(), compilationMessage.lineNum(), compilationMessage.linePos(), compilationMessage.offset(), compilationMessage.length() } };
}

RefPtr<WebCore::WebGPU::CompilationMessage> ConvertFromBackingContext::convertFromBacking(const CompilationMessage& compilationMessage)
{
    return WebCore::WebGPU::CompilationMessage::create(compilationMessage.message, compilationMessage.type, compilationMessage.lineNum, compilationMessage.linePos, compilationMessage.offset, compilationMessage.length);
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
