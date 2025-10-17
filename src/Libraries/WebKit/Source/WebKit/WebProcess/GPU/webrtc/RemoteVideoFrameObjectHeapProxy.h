/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include <wtf/Ref.h>
#include <wtf/ThreadSafeRefCounted.h>

#if PLATFORM(COCOA)
#include "RemoteVideoFrameObjectHeapProxyProcessor.h"
#endif

namespace WebKit {

class GPUProcessConnection;

#if PLATFORM(COCOA)
class RemoteVideoFrameProxy;
#endif

// Wrapper around RemoteVideoFrameObjectHeapProxyProcessor that will always be destroyed on main thread.
class RemoteVideoFrameObjectHeapProxy : public ThreadSafeRefCounted<RemoteVideoFrameObjectHeapProxy, WTF::DestructionThread::MainRunLoop> {
public:
    static Ref<RemoteVideoFrameObjectHeapProxy> create(GPUProcessConnection& connection) { return adoptRef(*new RemoteVideoFrameObjectHeapProxy(connection)); }
    ~RemoteVideoFrameObjectHeapProxy() = default;

#if PLATFORM(COCOA)
    Ref<RemoteVideoFrameObjectHeapProxyProcessor> protectedProcessor() { return m_processor; }

    void getVideoFrameBuffer(const RemoteVideoFrameProxy& proxy, bool canUseIOSurface, RemoteVideoFrameObjectHeapProxyProcessor::Callback&& callback) { protectedProcessor()->getVideoFrameBuffer(proxy, canUseIOSurface, WTFMove(callback)); }
    RefPtr<WebCore::NativeImage> getNativeImage(const WebCore::VideoFrame& frame) { return protectedProcessor()->getNativeImage(frame); }
#endif

private:
    explicit RemoteVideoFrameObjectHeapProxy(GPUProcessConnection& connection)
#if PLATFORM(COCOA)
        : m_processor(RemoteVideoFrameObjectHeapProxyProcessor::create(connection))
#endif
    {
    }
#if PLATFORM(COCOA)
    Ref<RemoteVideoFrameObjectHeapProxyProcessor> m_processor;
#endif
};

} // namespace WebKit

#endif
