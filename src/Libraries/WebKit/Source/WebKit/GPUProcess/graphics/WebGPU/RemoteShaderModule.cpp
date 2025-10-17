/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
#include "RemoteShaderModule.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteShaderModuleMessages.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUCompilationInfo.h>
#include <WebCore/WebGPUCompilationMessage.h>
#include <WebCore/WebGPUShaderModule.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteShaderModule);

RemoteShaderModule::RemoteShaderModule(WebCore::WebGPU::ShaderModule& shaderModule, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    : m_backing(shaderModule)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpu(gpu)
    , m_identifier(identifier)
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteShaderModule::messageReceiverName(), m_identifier.toUInt64());
}

RemoteShaderModule::~RemoteShaderModule() = default;

void RemoteShaderModule::destruct()
{
    protectedObjectHeap()->removeObject(m_identifier);
}

void RemoteShaderModule::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteShaderModule::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteShaderModule::compilationInfo(CompletionHandler<void(Vector<WebGPU::CompilationMessage>&&)>&& callback)
{
    protectedBacking()->compilationInfo([callback = WTFMove(callback)] (Ref<WebCore::WebGPU::CompilationInfo>&& compilationMessage) mutable {
        auto convertedMessages = compilationMessage->messages().map([] (const Ref<WebCore::WebGPU::CompilationMessage>& message) {
            return WebGPU::CompilationMessage {
                message->message(),
                message->type(),
                message->lineNum(),
                message->linePos(),
                message->offset(),
                message->length(),
            };
        });
        callback(WTFMove(convertedMessages));
    });
}

void RemoteShaderModule::setLabel(String&& label)
{
    protectedBacking()->setLabel(WTFMove(label));
}

Ref<WebCore::WebGPU::ShaderModule> RemoteShaderModule::protectedBacking()
{
    return m_backing;
}

Ref<IPC::StreamServerConnection> RemoteShaderModule::protectedStreamConnection() const
{
    return m_streamConnection;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
