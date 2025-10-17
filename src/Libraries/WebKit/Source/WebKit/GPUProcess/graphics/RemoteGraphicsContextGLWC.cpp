/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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

#if ENABLE(GPU_PROCESS) && ENABLE(WEBGL) && USE(GRAPHICS_LAYER_WC)
#include "GPUConnectionToWebProcess.h"
#include "WCContentBufferManager.h"

namespace WebKit {

class RemoteGraphicsContextGLWC final : public RemoteGraphicsContextGL {
public:
    RemoteGraphicsContextGLWC(GPUConnectionToWebProcess&, GraphicsContextGLIdentifier, RemoteRenderingBackend&, Ref<IPC::StreamServerConnection>&&);
    ~RemoteGraphicsContextGLWC() final = default;

    // RemoteGraphicsContextGL overrides.
    void platformWorkQueueInitialize(WebCore::GraphicsContextGLAttributes&&) final;
    void prepareForDisplay(CompletionHandler<void(std::optional<WCContentBufferIdentifier>)>&&) final;
};

Ref<RemoteGraphicsContextGL> RemoteGraphicsContextGL::create(GPUConnectionToWebProcess& gpuConnectionToWebProcess, WebCore::GraphicsContextGLAttributes&& attributes, GraphicsContextGLIdentifier graphicsContextGLIdentifier, RemoteRenderingBackend& renderingBackend, Ref<IPC::StreamServerConnection>&& streamConnection)
{
    auto instance = adoptRef(*new RemoteGraphicsContextGLWC(gpuConnectionToWebProcess, graphicsContextGLIdentifier, renderingBackend, WTFMove(streamConnection)));
    instance->initialize(WTFMove(attributes));
    return instance;
}

RemoteGraphicsContextGLWC::RemoteGraphicsContextGLWC(GPUConnectionToWebProcess& gpuConnectionToWebProcess, GraphicsContextGLIdentifier graphicsContextGLIdentifier, RemoteRenderingBackend& renderingBackend, Ref<IPC::StreamServerConnection>&& streamConnection)
    : RemoteGraphicsContextGL(gpuConnectionToWebProcess, graphicsContextGLIdentifier, renderingBackend, WTFMove(streamConnection))
{
}

void RemoteGraphicsContextGLWC::platformWorkQueueInitialize(WebCore::GraphicsContextGLAttributes&& attributes)
{
    m_context = GCGLContext::create(WTFMove(attributes));
}

void RemoteGraphicsContextGLWC::prepareForDisplay(CompletionHandler<void(std::optional<WCContentBufferIdentifier>)>&& completionHandler)
{
    m_context->prepareForDisplay();
    auto identifier = WCContentBufferManager::singleton().acquireContentBufferIdentifier(m_webProcessIdentifier, m_context->layerContentsDisplayDelegate()->platformLayer());
    completionHandler(identifier);
}

}

#endif // ENABLE(GPU_PROCESS) && ENABLE(WEBGL) && USE(GRAPHICS_LAYER_WC)
