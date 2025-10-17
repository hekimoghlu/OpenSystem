/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#include "TrackQueue.h"

#if ENABLE(MEDIA_SOURCE) && USE(GSTREAMER)

GST_DEBUG_CATEGORY_STATIC(webkit_mse_track_queue_debug);
#define GST_CAT_DEFAULT webkit_mse_track_queue_debug

namespace WebCore {

TrackQueue::TrackQueue(TrackID trackId)
    : m_trackId(trackId)
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_mse_track_queue_debug, "webkitmsetrackqueue", 0, "WebKit MSE TrackQueue");
    });
}

void TrackQueue::enqueueObject(GRefPtr<GstMiniObject>&& object)
{
    ASSERT(isMainThread());
    ASSERT(GST_IS_SAMPLE(object.get()) || GST_IS_EVENT(object.get()));

    if (GST_IS_SAMPLE(object.get())) {
        GST_TRACE("TrackQueue for '%" PRIu64 "': Putting object sample in the queue: %" GST_PTR_FORMAT " Buffer: %" GST_PTR_FORMAT ". notEmptyCallback currently %s.",
            m_trackId, object.get(), gst_sample_get_buffer(GST_SAMPLE(object.get())),
            m_notEmptyCallback ? "set, will be called" : "unset");
    } else {
        GST_DEBUG("TrackQueue for '%" PRIu64 "': Putting object event in the queue: %" GST_PTR_FORMAT ". notEmptyCallback currently %s.",
            m_trackId, object.get(),
            m_notEmptyCallback ? "set, will be called" : "unset");
    }
    if (!m_notEmptyCallback)
        m_queue.append(WTFMove(object));
    else {
        // If a low level callback was ever set, it had to be dispatched when the queue was empty at latest.
        ASSERT(!m_lowLevelCallback);

        NotEmptyHandler notEmptyCallback;
        std::swap(notEmptyCallback, m_notEmptyCallback);
        notEmptyCallback(WTFMove(object));
    }
}

void TrackQueue::clear()
{
    ASSERT(isMainThread());
    m_queue.clear();
    GST_DEBUG("TrackQueue for '%" PRIu64 "': Emptied.", m_trackId);
    // Notify main thread of low level reached if it proceeds.
    checkLowLevel();
}

void TrackQueue::flush()
{
    clear();
    // If there was a callback in the streaming thread waiting for a sample to be added, cancel it.
    if (m_notEmptyCallback) {
        m_notEmptyCallback = nullptr;
        GST_DEBUG("TrackQueue for '%" PRIu64 "': notEmptyCallback unset.", m_trackId);
    }
}

void TrackQueue::notifyWhenLowLevel(LowLevelHandler&& lowLevelCallback)
{
    ASSERT(isMainThread());
    GST_TRACE("TrackQueue for '%" PRIu64 "': Setting lowLevelCallback%s.", m_trackId,
        m_lowLevelCallback ? " (previous callback will be discarded)" : "");
    m_lowLevelCallback = WTFMove(lowLevelCallback);
    checkLowLevel();
}

GRefPtr<GstMiniObject> TrackQueue::pop()
{
    ASSERT(!isEmpty());
    GRefPtr<GstMiniObject> object = m_queue.takeFirst();
    if (GST_IS_SAMPLE(object.get())) {
        GST_TRACE("TrackQueue for '%" PRIu64 "': Popped object sample from the queue: %" GST_PTR_FORMAT " Buffer: %" GST_PTR_FORMAT,
            m_trackId, object.get(), gst_sample_get_buffer(GST_SAMPLE(object.get())));
    } else {
        GST_DEBUG("TrackQueue for '%" PRIu64 "': Popped object event from the queue: %" GST_PTR_FORMAT,
            m_trackId, object.get());
    }
    checkLowLevel();
    return object;
}

void TrackQueue::notifyWhenNotEmpty(NotEmptyHandler&& notEmptyCallback)
{
    ASSERT(!isMainThread());
    ASSERT(!m_notEmptyCallback);
    m_notEmptyCallback = WTFMove(notEmptyCallback);
    GST_TRACE("TrackQueue for '%" PRIu64 "': notEmptyCallback set.", m_trackId);
}

void TrackQueue::resetNotEmptyHandler()
{
    ASSERT(!isMainThread());
    if (!m_notEmptyCallback)
        return;
    m_notEmptyCallback = nullptr;
    GST_TRACE("TrackQueue for '%" PRIu64 "': notEmptyCallback reset.", m_trackId);
}

void TrackQueue::checkLowLevel()
{
    if (!m_lowLevelCallback || durationEnqueued() > s_durationEnqueuedLowWaterLevel)
        return;

    LowLevelHandler lowLevelCallback;
    std::swap(lowLevelCallback, m_lowLevelCallback);
    GST_TRACE("TrackQueue for '%" PRIu64 "': lowLevelCallback called.", m_trackId);
    lowLevelCallback();
}

GstClockTime TrackQueue::durationEnqueued() const
{
    // Find the first and last GstSample in the queue and subtract their DTS.

    auto frontIter = std::find_if(m_queue.begin(), m_queue.end(), [](const GRefPtr<GstMiniObject>& object) {
        return GST_IS_SAMPLE(object.get());
    });

    // If there are no samples in the queue, that makes total duration of enqueued frames of zero.
    if (frontIter == m_queue.end())
        return 0;

    auto backIter = std::find_if(m_queue.rbegin(), m_queue.rend(), [](const GRefPtr<GstMiniObject>& object) {
        return GST_IS_SAMPLE(object.get());
    });

    const GstBuffer* front = gst_sample_get_buffer(GST_SAMPLE(frontIter->get()));
    const GstBuffer* back = gst_sample_get_buffer(GST_SAMPLE(backIter->get()));
    return GST_BUFFER_DTS_OR_PTS(back) - GST_BUFFER_DTS_OR_PTS(front);
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE) && USE(GSTREAMER)
