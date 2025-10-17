/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
#include "WebGPUShaderModuleImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUCompilationInfo.h"
#include "WebGPUCompilationMessage.h"
#include "WebGPUCompilationMessageType.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebGPU/WebGPUExt.h>

#include <wtf/BlockPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ShaderModuleImpl);

ShaderModuleImpl::ShaderModuleImpl(WebGPUPtr<WGPUShaderModule>&& shaderModule, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(shaderModule))
    , m_convertToBackingContext(convertToBackingContext)
{
}

ShaderModuleImpl::~ShaderModuleImpl() = default;

static CompilationMessageType convertFromBacking(WGPUCompilationMessageType type)
{
    switch (type) {
    case WGPUCompilationMessageType_Error:
        return CompilationMessageType::Error;
    case WGPUCompilationMessageType_Warning:
        return CompilationMessageType::Warning;
    case WGPUCompilationMessageType_Info:
        return CompilationMessageType::Info;
    case WGPUCompilationMessageType_Force32:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

static void compilationInfoCallback(WGPUCompilationInfoRequestStatus status, const WGPUCompilationInfo* compilationInfo, void* userdata)
{
    auto block = reinterpret_cast<void(^)(WGPUCompilationInfoRequestStatus, const WGPUCompilationInfo*)>(userdata);
    block(status, compilationInfo);
    Block_release(block); // Block_release is matched with Block_copy below in AdapterImpl::requestDevice().
}

void ShaderModuleImpl::compilationInfo(CompletionHandler<void(Ref<CompilationInfo>&&)>&& callback)
{
    auto blockPtr = makeBlockPtr([callback = WTFMove(callback)](WGPUCompilationInfoRequestStatus, const WGPUCompilationInfo* compilationInfo) mutable {
        Vector<Ref<CompilationMessage>> messages;
        if (!compilationInfo || !compilationInfo->messageCount) {
            callback(CompilationInfo::create(WTFMove(messages)));
            return;
        }

        for (size_t i = 0; i < compilationInfo->messageCount; ++i) {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
            auto& message = compilationInfo->messages[i];
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
            messages.append(CompilationMessage::create(message.message, convertFromBacking(message.type), message.lineNum, message.linePos + 1, message.offset, message.length));
        }

        callback(CompilationInfo::create(WTFMove(messages)));
    });

    wgpuShaderModuleGetCompilationInfo(m_backing.get(), &compilationInfoCallback, Block_copy(blockPtr.get()));
}

void ShaderModuleImpl::setLabelInternal(const String& label)
{
    wgpuShaderModuleSetLabel(m_backing.get(), label.utf8().data());
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
