/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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
#include "RemoteGraphicsContextGL.h"

#if ENABLE(GPU_PROCESS) && ENABLE(WEBGL) && USE(COORDINATED_GRAPHICS) && USE(GBM)
#include <WebCore/GLFence.h>
#include <epoxy/gl.h>

namespace WebKit {

class RemoteGraphicsContextGLGBM final : public RemoteGraphicsContextGL {
public:
    RemoteGraphicsContextGLGBM(GPUConnectionToWebProcess&, GraphicsContextGLIdentifier, RemoteRenderingBackend&, Ref<IPC::StreamServerConnection>&&);

private:
    void platformWorkQueueInitialize(WebCore::GraphicsContextGLAttributes&&) final;
    void prepareForDisplay(CompletionHandler<void(uint64_t, std::optional<WebCore::DMABufBuffer::Attributes>&&, UnixFileDescriptor&&)>&&) final;
};

RemoteGraphicsContextGLGBM::RemoteGraphicsContextGLGBM(GPUConnectionToWebProcess& connection, GraphicsContextGLIdentifier identifier, RemoteRenderingBackend& renderingBackend, Ref<IPC::StreamServerConnection>&& streamConnection)
    : RemoteGraphicsContextGL(connection, identifier, renderingBackend, WTFMove(streamConnection))
{ }

void RemoteGraphicsContextGLGBM::platformWorkQueueInitialize(WebCore::GraphicsContextGLAttributes&& attributes)
{
    assertIsCurrent(workQueue());
    m_context = WebCore::GraphicsContextGLTextureMapperGBM::create(WTFMove(attributes));
}

void RemoteGraphicsContextGLGBM::prepareForDisplay(CompletionHandler<void(uint64_t, std::optional<WebCore::DMABufBuffer::Attributes>&&, UnixFileDescriptor&&)>&& completionHandler)
{
    assertIsCurrent(workQueue());
    std::unique_ptr<WebCore::GLFence> fence;
    m_context->prepareForDisplayWithFinishedSignal([&fence] {
        fence = WebCore::GLFence::createExportable();
        if (fence)
            return;

        fence = WebCore::GLFence::create();
    });

    auto* buffer = m_context->displayBuffer();
    if (!buffer) {
        completionHandler(0, std::nullopt, { });
        return;
    }

    UnixFileDescriptor fenceFD;
    if (fence) {
        fenceFD = fence->exportFD();
        if (!fenceFD)
            fence->clientWait();
    } else
        glFlush();

    completionHandler(buffer->id(), buffer->takeAttributes(), WTFMove(fenceFD));
}

Ref<RemoteGraphicsContextGL> RemoteGraphicsContextGL::create(GPUConnectionToWebProcess& connection, WebCore::GraphicsContextGLAttributes&& attributes, GraphicsContextGLIdentifier identifier, RemoteRenderingBackend& renderingBackend, Ref<IPC::StreamServerConnection>&& streamConnection)
{
    auto instance = adoptRef(*new RemoteGraphicsContextGLGBM(connection, identifier, renderingBackend, WTFMove(streamConnection)));
    instance->initialize(WTFMove(attributes));
    return instance;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(WEBGL) && USE(COORDINATED_GRAPHICS) && USE(GBM)
