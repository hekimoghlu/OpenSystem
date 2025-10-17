/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#import "RTCVideoFrameReorderQueue.h"

namespace webrtc {

bool RTCVideoFrameReorderQueue::isEmpty()
{
    return _reorderQueue.empty();
}

uint8_t RTCVideoFrameReorderQueue::reorderSize() const
{
    webrtc::MutexLock lock(&_reorderQueueLock);
    return _reorderSize;
}

void RTCVideoFrameReorderQueue::setReorderSize(uint8_t size)
{
    webrtc::MutexLock lock(&_reorderQueueLock);
    _reorderSize = size;
}

void RTCVideoFrameReorderQueue::append(RTCVideoFrame* frame, uint8_t reorderSize)
{
    webrtc::MutexLock lock(&_reorderQueueLock);
    _reorderQueue.push_back(std::make_unique<RTCVideoFrameWithOrder>(frame, reorderSize));
    std::sort(_reorderQueue.begin(), _reorderQueue.end(), [](auto& a, auto& b) {
        return a->timeStamp < b->timeStamp;
    });
}

RTCVideoFrame* RTCVideoFrameReorderQueue::takeIfAvailable(bool& moreFramesAvailable)
{
    moreFramesAvailable = false;
    auto areFramesAvailable = [&] -> bool { return _reorderQueue.size() && _reorderQueue.size() > _reorderQueue.front()->reorderSize; };

    webrtc::MutexLock lock(&_reorderQueueLock);
    if (areFramesAvailable()) {
        auto *frame = _reorderQueue.front()->take();
        _reorderQueue.pop_front();
        moreFramesAvailable = areFramesAvailable();
        return frame;
    }

    return nil;
}

RTCVideoFrame* RTCVideoFrameReorderQueue::takeIfAny()
{
    webrtc::MutexLock lock(&_reorderQueueLock);
    if (_reorderQueue.size()) {
        auto *frame = _reorderQueue.front()->take();
        _reorderQueue.pop_front();
        return frame;
    }
    return nil;
}

}
