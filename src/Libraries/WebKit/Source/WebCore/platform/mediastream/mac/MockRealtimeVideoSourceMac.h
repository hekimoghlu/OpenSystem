/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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

#if ENABLE(MEDIA_STREAM)

#include "MockRealtimeVideoSource.h"
#include "PixelBufferConformerCV.h"
#include <wtf/WorkQueue.h>

typedef struct __CVBuffer *CVBufferRef;
typedef CVBufferRef CVImageBufferRef;
typedef CVImageBufferRef CVPixelBufferRef;
typedef struct __CVPixelBufferPool *CVPixelBufferPoolRef;

namespace WebCore {

class ImageTransferSessionVT;

class MockRealtimeVideoSourceMac final : public MockRealtimeVideoSource {
public:
    static Ref<MockRealtimeVideoSource> createForMockDisplayCapturer(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&&, std::optional<PageIdentifier>);
    static Ref<MockRealtimeVideoSource> create(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&& salt, std::optional<PageIdentifier> pageIdentifier) { return adoptRef(*new MockRealtimeVideoSourceMac(WTFMove(deviceID), WTFMove(name), WTFMove(salt), WTFMove(pageIdentifier))); }

    ~MockRealtimeVideoSourceMac();

private:
    MockRealtimeVideoSourceMac(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&&, std::optional<PageIdentifier>);

    PlatformLayer* platformLayer() const;
    void updateSampleBuffer() final;
    bool canResizeVideoFrames() const final { return true; }

    std::unique_ptr<ImageTransferSessionVT> m_imageTransferSession;
    IntSize m_presetSize;
    size_t m_pixelGenerationFailureCount { 0 };
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
