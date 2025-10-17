/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#include "RemoteShaderModuleProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteShaderModuleMessages.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUCompilationInfo.h>
#include <WebCore/WebGPUCompilationMessage.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteShaderModuleProxy);

RemoteShaderModuleProxy::RemoteShaderModuleProxy(RemoteDeviceProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_parent(parent)
{
}

RemoteShaderModuleProxy::~RemoteShaderModuleProxy()
{
    auto sendResult = send(Messages::RemoteShaderModule::Destruct());
    UNUSED_VARIABLE(sendResult);
}

void RemoteShaderModuleProxy::compilationInfo(CompletionHandler<void(Ref<WebCore::WebGPU::CompilationInfo>&&)>&& callback)
{
    auto sendResult = sendWithAsyncReply(Messages::RemoteShaderModule::CompilationInfo(), [callback = WTFMove(callback)](auto messages) mutable {
        auto backingMessages = messages.map([](CompilationMessage compilationMessage) {
            return WebCore::WebGPU::CompilationMessage::create(WTFMove(compilationMessage.message), compilationMessage.type, compilationMessage.lineNum, compilationMessage.linePos, compilationMessage.offset, compilationMessage.length);
        });
        callback(WebCore::WebGPU::CompilationInfo::create(WTFMove(backingMessages)));
    });

    UNUSED_PARAM(sendResult);
}

void RemoteShaderModuleProxy::setLabelInternal(const String& label)
{
    auto sendResult = send(Messages::RemoteShaderModule::SetLabel(label));
    UNUSED_VARIABLE(sendResult);
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
