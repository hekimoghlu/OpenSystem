/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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

#include "DecodingOptions.h"
#include "ImageTypes.h"
#include <wtf/SynchronizedFixedQueue.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/WorkQueue.h>

namespace WebCore {

class BitmapImageSource;

class ImageFrameWorkQueue : public ThreadSafeRefCounted<ImageFrameWorkQueue> {
public:
    struct Request {
        unsigned index;
        SubsamplingLevel subsamplingLevel;
        ImageAnimatingState animatingState;
        DecodingOptions options;
        friend bool operator==(const Request&, const Request&) = default;
    };

    static Ref<ImageFrameWorkQueue> create(BitmapImageSource&);

    void start();
    void dispatch(const Request&);
    void stop();

    bool isIdle() const { return m_decodeQueue.isEmpty(); }
    bool isPendingDecodingAtIndex(unsigned index, SubsamplingLevel, const DecodingOptions&) const;

    void setMinimumDecodingDurationForTesting(Seconds duration) { m_minimumDecodingDurationForTesting = duration; }
    void dump(TextStream&) const;

private:
    ImageFrameWorkQueue(BitmapImageSource&);

    Ref<BitmapImageSource> protectedSource() const { return m_source.get().releaseNonNull(); }

    static const int BufferSize = 8;
    using RequestQueue = SynchronizedFixedQueue<Request, BufferSize>;
    using DecodeQueue = Deque<Request, BufferSize>;

    RequestQueue& requestQueue();
    DecodeQueue& decodeQueue() { return m_decodeQueue; }

    Seconds minimumDecodingDurationForTesting() const { return m_minimumDecodingDurationForTesting; }

    ThreadSafeWeakPtr<BitmapImageSource> m_source;

    RefPtr<RequestQueue> m_requestQueue;
    DecodeQueue m_decodeQueue;
    RefPtr<WorkQueue> m_workQueue;

    Seconds m_minimumDecodingDurationForTesting;
};

} // namespace WebCore
