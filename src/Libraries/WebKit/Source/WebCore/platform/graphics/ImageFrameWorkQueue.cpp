/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#include "ImageFrameWorkQueue.h"

#include "BitmapImageSource.h"
#include "ImageDecoder.h"
#include "Logging.h"
#include <wtf/SystemTracing.h>

namespace WebCore {

Ref<ImageFrameWorkQueue> ImageFrameWorkQueue::create(BitmapImageSource& source)
{
    return adoptRef(*new ImageFrameWorkQueue(source));
}

ImageFrameWorkQueue::ImageFrameWorkQueue(BitmapImageSource& source)
    : m_source(source)
{
}

ImageFrameWorkQueue::RequestQueue& ImageFrameWorkQueue::requestQueue()
{
    if (!m_requestQueue)
        m_requestQueue = RequestQueue::create();

    return *m_requestQueue;
}

void ImageFrameWorkQueue::start()
{
    ASSERT(isMainThread());

    if (m_workQueue)
        return;

    RefPtr decoder = protectedSource()->decoder();
    if (!decoder)
        return;

    m_workQueue = WorkQueue::create("org.webkit.ImageDecoder"_s, WorkQueue::QOS::Default);

    m_workQueue->dispatch([protectedThis = Ref { *this }, protectedWorkQueue = Ref { *m_workQueue }, protectedSource = this->protectedSource(), protectedDecoder = Ref { *decoder }, protectedRequestQueue = Ref { requestQueue() }] () mutable {
        Request request;
        while (protectedRequestQueue->dequeue(request)) {
            TraceScope tracingScope(AsyncImageDecodeStart, AsyncImageDecodeEnd);

            auto minimumDecodingDuration = protectedThis->minimumDecodingDurationForTesting();

            MonotonicTime startingTime;
            if (minimumDecodingDuration > 0_s)
                startingTime = MonotonicTime::now();

            PlatformImagePtr platformImage = protectedDecoder->createFrameImageAtIndex(request.index, request.subsamplingLevel, request.options);
            RefPtr nativeImage = NativeImage::create(WTFMove(platformImage));

            // Pretend as if decoding the frame took minimumDecodingDuration.
            if (minimumDecodingDuration > 0_s) {
                auto actualDecodingDuration = MonotonicTime::now() - startingTime;
                if (minimumDecodingDuration > actualDecodingDuration)
                    sleep(minimumDecodingDuration - actualDecodingDuration);
            }

            // Even if we fail to decode the frame, it is important to sync the main thread with this result.
            callOnMainThread([protectedThis, protectedWorkQueue, protectedSource, request, nativeImage = WTFMove(nativeImage)] () mutable {
                // The WorkQueue may have been recreated before the frame was decoded.
                if (protectedWorkQueue.ptr() != protectedThis->m_workQueue || protectedSource.ptr() != protectedThis->m_source.get()) {
                    LOG(Images, "ImageFrameWorkQueue::%s - %p - url: %s. WorkQueue was recreated at index = %d.", __FUNCTION__, protectedThis.ptr(), protectedSource->sourceUTF8().data(), request.index);
                    return;
                }

                // The DecodeQueue may have been cleared before the frame was decoded.
                if (protectedThis->decodeQueue().isEmpty() || protectedThis->decodeQueue().first() != request) {
                    LOG(Images, "ImageFrameWorkQueue::%s - %p - url: %s. DecodeQueue was cleared at index = %d.", __FUNCTION__, protectedThis.ptr(), protectedSource->sourceUTF8().data(), request.index);
                    return;
                }

                protectedThis->decodeQueue().removeFirst();
                protectedSource->imageFrameDecodeAtIndexHasFinished(request.index, request.subsamplingLevel, request.animatingState, request.options, WTFMove(nativeImage));
            });
        }

        // Ensure destruction happens on creation thread.
        callOnMainThread([protectedThis = WTFMove(protectedThis), protectedWorkQueue = WTFMove(protectedWorkQueue), protectedSource = WTFMove(protectedSource)] () mutable { });
    });
}

void ImageFrameWorkQueue::dispatch(const Request& request)
{
    ASSERT(isMainThread());

    requestQueue().enqueue(request);
    decodeQueue().append(request);

    start();
}

void ImageFrameWorkQueue::stop()
{
    ASSERT(isMainThread());

    Ref source = protectedSource();

    for (auto& request : m_decodeQueue) {
        LOG(Images, "ImageFrameWorkQueue::%s - %p - url: %s. Decoding was cancelled for frame at index = %d.", __FUNCTION__, this, source->sourceUTF8().data(), request.index);
        source->destroyNativeImageAtIndex(request.index);
    }

    if (m_requestQueue) {
        m_requestQueue->close();
        m_requestQueue = nullptr;
    }

    m_decodeQueue.clear();
    m_workQueue = nullptr;
}

bool ImageFrameWorkQueue::isPendingDecodingAtIndex(unsigned index, SubsamplingLevel subsamplingLevel, const DecodingOptions& options) const
{
    ASSERT(isMainThread());

    auto it = std::find_if(m_decodeQueue.begin(), m_decodeQueue.end(), [index, subsamplingLevel, &options](const Request& request) {
        return request.index == index && subsamplingLevel >= request.subsamplingLevel && request.options.isCompatibleWith(options);
    });
    return it != m_decodeQueue.end();
}

void ImageFrameWorkQueue::dump(TextStream& ts) const
{
    if (isIdle())
        return;

    ts.dumpProperty("pending-for-decoding", m_decodeQueue.size());
}

} // namespace WebCore
