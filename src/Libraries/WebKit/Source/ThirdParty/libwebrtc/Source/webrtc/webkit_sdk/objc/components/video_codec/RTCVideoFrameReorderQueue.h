/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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
#import "base/RTCVideoFrame.h"
#include <deque>
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

class RTCVideoFrameReorderQueue {
public:
    RTCVideoFrameReorderQueue() = default;

    struct RTCVideoFrameWithOrder {
        RTCVideoFrameWithOrder(RTCVideoFrame* frame, uint64_t reorderSize)
            : frame((__bridge_retained void*)frame)
            , timeStamp(frame.timeStamp)
            , reorderSize(reorderSize)
        {
        }

        ~RTCVideoFrameWithOrder()
        {
            if (frame)
                take();
        }

        RTCVideoFrame* take()
        {
            auto* rtcFrame = (__bridge_transfer RTCVideoFrame *)frame;
            frame = nullptr;
            return rtcFrame;
        }

        void* frame;
        uint64_t timeStamp;
        uint64_t reorderSize;
    };

    bool isEmpty();
    uint8_t reorderSize() const;
    void setReorderSize(uint8_t);
    void append(RTCVideoFrame*, uint8_t);
    RTCVideoFrame *takeIfAvailable(bool& moreFramesAvailable);
    RTCVideoFrame *takeIfAny();

private:
    std::deque<std::unique_ptr<RTCVideoFrameWithOrder>> _reorderQueue;
    uint8_t _reorderSize { 0 };
    mutable webrtc::Mutex _reorderQueueLock;
};

}
