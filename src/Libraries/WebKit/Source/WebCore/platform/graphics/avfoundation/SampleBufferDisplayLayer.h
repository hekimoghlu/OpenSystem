/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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

#include "PlatformLayer.h"
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/CompletionHandler.h>
#include <wtf/MachSendRight.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WTF {
class MediaTime;
}

namespace WebCore {
class IntSize;
class VideoFrame;

enum class VideoFrameRotation : uint16_t;

using LayerHostingContextID = uint32_t;

class SampleBufferDisplayLayerClient : public AbstractRefCountedAndCanMakeWeakPtr<SampleBufferDisplayLayerClient> {
public:
    virtual ~SampleBufferDisplayLayerClient() = default;
    virtual void sampleBufferDisplayLayerStatusDidFail() = 0;
#if PLATFORM(IOS_FAMILY)
    virtual bool canShowWhileLocked() const = 0;
#endif
};

class SampleBufferDisplayLayer : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<SampleBufferDisplayLayer, WTF::DestructionThread::MainRunLoop> {
public:
    WEBCORE_EXPORT static RefPtr<SampleBufferDisplayLayer> create(SampleBufferDisplayLayerClient&);
    using LayerCreator = RefPtr<SampleBufferDisplayLayer> (*)(SampleBufferDisplayLayerClient&);
    WEBCORE_EXPORT static void setCreator(LayerCreator);

    virtual ~SampleBufferDisplayLayer() = default;

    virtual void initialize(bool hideRootLayer, IntSize, bool shouldMaintainAspectRatio, CompletionHandler<void(bool didSucceed)>&&) = 0;
#if !RELEASE_LOG_DISABLED
    virtual void setLogIdentifier(String&&) = 0;
#endif
    virtual bool didFail() const = 0;

    virtual void updateDisplayMode(bool hideDisplayLayer, bool hideRootLayer) = 0;

    virtual void updateBoundsAndPosition(CGRect, std::optional<WTF::MachSendRight>&& = std::nullopt) = 0;

    virtual void flush() = 0;
    virtual void flushAndRemoveImage() = 0;

    virtual void play() = 0;
    virtual void pause() = 0;

    virtual void enqueueBlackFrameFrom(const VideoFrame&) { };
    virtual void enqueueVideoFrame(VideoFrame&) = 0;
    virtual void clearVideoFrames() = 0;

    virtual PlatformLayer* rootLayer() = 0;
    virtual void setShouldMaintainAspectRatio(bool) = 0;

    enum class RenderPolicy { TimingInfo, Immediately };
    virtual void setRenderPolicy(RenderPolicy) { };

    virtual LayerHostingContextID hostingContextID() const { return 0; }

protected:
    explicit SampleBufferDisplayLayer(SampleBufferDisplayLayerClient&);

    bool canShowWhileLocked();
    WeakPtr<SampleBufferDisplayLayerClient> m_client;

private:
    static LayerCreator m_layerCreator;
};

inline SampleBufferDisplayLayer::SampleBufferDisplayLayer(SampleBufferDisplayLayerClient& client)
    : m_client(client)
{
}

inline bool SampleBufferDisplayLayer::canShowWhileLocked()
{
#if PLATFORM(IOS_FAMILY)
    return m_client && m_client->canShowWhileLocked();
#else
    return false;
#endif
}

}
