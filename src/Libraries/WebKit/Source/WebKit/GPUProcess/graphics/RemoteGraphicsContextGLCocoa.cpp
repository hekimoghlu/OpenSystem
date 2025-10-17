/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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

#if ENABLE(GPU_PROCESS) && ENABLE(WEBGL) && PLATFORM(COCOA)

#include "GPUConnectionToWebProcess.h"
#include "IPCUtilities.h"
#include "RemoteSharedResourceCache.h"
#include <WebCore/ProcessIdentity.h>
#include <wtf/MachSendRight.h>

#if ENABLE(VIDEO)
#include "RemoteVideoFrameObjectHeap.h"
#include <WebCore/GraphicsContextGLCV.h>
#include <WebCore/MediaSampleAVFObjC.h>
#include <WebCore/VideoFrameCV.h>
#endif

namespace WebKit {

#if ENABLE(VIDEO)
void RemoteGraphicsContextGL::copyTextureFromVideoFrame(WebKit::SharedVideoFrame&& frame, PlatformGLObject texture, uint32_t target, int32_t level, uint32_t internalFormat, uint32_t format, uint32_t type, bool premultiplyAlpha, bool flipY, CompletionHandler<void(bool)>&& completionHandler)
{
    assertIsCurrent(workQueue());
    RefPtr videoFrame = m_sharedVideoFrameReader.read(WTFMove(frame));
    if (!videoFrame) {
        ASSERT_IS_TESTING_IPC();
        completionHandler(false);
        return;
    }
    if (!m_objectNames.isValidKey(texture)) {
        ASSERT_IS_TESTING_IPC();
        return;
    }
    texture = m_objectNames.get(texture);
    bool result = protectedContext()->copyTextureFromVideoFrame(*videoFrame, texture, target, level, internalFormat, format, type, premultiplyAlpha, flipY);
    completionHandler(result);
}

void RemoteGraphicsContextGL::setSharedVideoFrameSemaphore(IPC::Semaphore&& semaphore)
{
    m_sharedVideoFrameReader.setSemaphore(WTFMove(semaphore));
}

void RemoteGraphicsContextGL::setSharedVideoFrameMemory(SharedMemory::Handle&& handle)
{
    m_sharedVideoFrameReader.setSharedMemory(WTFMove(handle));
}
#endif

namespace {

class RemoteGraphicsContextGLCocoa final : public RemoteGraphicsContextGL {
public:
    RemoteGraphicsContextGLCocoa(GPUConnectionToWebProcess&, GraphicsContextGLIdentifier, RemoteRenderingBackend&, Ref<IPC::StreamServerConnection>&&);
    ~RemoteGraphicsContextGLCocoa() final = default;

    // RemoteGraphicsContextGL overrides.
    void platformWorkQueueInitialize(WebCore::GraphicsContextGLAttributes&&) final;
    void prepareForDisplay(IPC::Semaphore&&, CompletionHandler<void(WTF::MachSendRight&&)>&&) final;
private:
};

}

Ref<RemoteGraphicsContextGL> RemoteGraphicsContextGL::create(GPUConnectionToWebProcess& gpuConnectionToWebProcess, WebCore::GraphicsContextGLAttributes&& attributes, GraphicsContextGLIdentifier graphicsContextGLIdentifier, RemoteRenderingBackend& renderingBackend, Ref<IPC::StreamServerConnection>&& streamConnection)
{
    auto instance = adoptRef(*new RemoteGraphicsContextGLCocoa(gpuConnectionToWebProcess, graphicsContextGLIdentifier, renderingBackend, WTFMove(streamConnection)));
    instance->initialize(WTFMove(attributes));
    return instance;
}

RemoteGraphicsContextGLCocoa::RemoteGraphicsContextGLCocoa(GPUConnectionToWebProcess& gpuConnectionToWebProcess, GraphicsContextGLIdentifier graphicsContextGLIdentifier, RemoteRenderingBackend& renderingBackend, Ref<IPC::StreamServerConnection>&& streamConnection)
    : RemoteGraphicsContextGL(gpuConnectionToWebProcess, graphicsContextGLIdentifier, renderingBackend, WTFMove(streamConnection))
{
}

void RemoteGraphicsContextGLCocoa::platformWorkQueueInitialize(WebCore::GraphicsContextGLAttributes&& attributes)
{
    assertIsCurrent(workQueue());
    m_context = WebCore::GraphicsContextGLCocoa::create(WTFMove(attributes), WebCore::ProcessIdentity { m_sharedResourceCache->resourceOwner() });
}

void RemoteGraphicsContextGLCocoa::prepareForDisplay(IPC::Semaphore&& finishedSemaphore, CompletionHandler<void(WTF::MachSendRight&&)>&& completionHandler)
{
    assertIsCurrent(workQueue());
    RefPtr context = m_context;

    context->prepareForDisplayWithFinishedSignal([finishedSemaphore = WTFMove(finishedSemaphore)]() mutable {
        finishedSemaphore.signal();
    });
    MachSendRight sendRight;
    if (WebCore::IOSurface* surface = context->displayBufferSurface())
        sendRight = surface->createSendRight();
    completionHandler(WTFMove(sendRight));
}

}

#endif
