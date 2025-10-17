/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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

#include "Connection.h"
#include "RemoteVideoFrameObjectHeapProxy.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WorkerClient.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class Page;
}

namespace WebCore::WebGPU {
class GPU;
}

namespace WebKit {

class WebPage;

class WebWorkerClient : public WebCore::WorkerClient {
    WTF_MAKE_TZONE_ALLOCATED(WebWorkerClient);
public:
    ~WebWorkerClient();
    // Constructed on the main thread, and then transferred to the
    // worker thread. All further operations on this object will
    // happen on the worker.
    // Any details needed from the page must be copied at this
    // point, but can't hold references to any main-thread objects.
    static UniqueRef<WebWorkerClient> create(WebCore::Page&, SerialFunctionDispatcher&);

    UniqueRef<WorkerClient> createNestedWorkerClient(SerialFunctionDispatcher&) override;

    WebCore::PlatformDisplayID displayID() const final;

    RefPtr<WebCore::ImageBuffer> sinkIntoImageBuffer(std::unique_ptr<WebCore::SerializedImageBuffer>) override;
    RefPtr<WebCore::ImageBuffer> createImageBuffer(const WebCore::FloatSize&, WebCore::RenderingMode, WebCore::RenderingPurpose, float resolutionScale, const WebCore::DestinationColorSpace&, WebCore::ImageBufferPixelFormat) const override;
#if ENABLE(WEBGL)
    RefPtr<WebCore::GraphicsContextGL> createGraphicsContextGL(const WebCore::GraphicsContextGLAttributes&) const override;
#endif

#if HAVE(WEBGPU_IMPLEMENTATION)
    RefPtr<WebCore::WebGPU::GPU> createGPUForWebGPU() const override;
#endif

protected:
    WebWorkerClient(SerialFunctionDispatcher&, WebCore::PlatformDisplayID);

    // m_dispatcher should stay alive as long as WebWorkerClient is alive.
    RefPtr<SerialFunctionDispatcher> dispatcher() const { return m_dispatcher.get(); }

    ThreadSafeWeakPtr<SerialFunctionDispatcher> m_dispatcher;
    const WebCore::PlatformDisplayID m_displayID;
};

} // namespace WebKit
