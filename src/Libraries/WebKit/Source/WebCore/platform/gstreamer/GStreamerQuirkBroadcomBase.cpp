/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#include "GStreamerQuirkBroadcomBase.h"

#include <string.h>

#if USE(GSTREAMER)

#include "GStreamerCommon.h"

namespace WebCore {

GST_DEBUG_CATEGORY_STATIC(webkit_broadcom_base_quirks_debug);
#define GST_CAT_DEFAULT webkit_broadcom_base_quirks_debug

GStreamerQuirkBroadcomBase::GStreamerQuirkBroadcomBase()
{
    GST_DEBUG_CATEGORY_INIT(webkit_broadcom_base_quirks_debug, "webkitquirksbroadcombase", 0, "WebKit Broadcom Base Quirks");
}

ASCIILiteral GStreamerQuirkBroadcomBase::queryBufferingPercentage(MediaPlayerPrivateGStreamer* playerPrivate, const GRefPtr<GstQuery>& query) const
{
    auto& state = ensureState(playerPrivate);

    if (playerPrivate->shouldDownload() || !state.m_queue2
        || !gst_element_query(state.m_queue2.get(), query.get()))
        return nullptr;
    return "queue2"_s;
}

int GStreamerQuirkBroadcomBase::correctBufferingPercentage(MediaPlayerPrivateGStreamer* playerPrivate, int originalBufferingPercentage, GstBufferingMode mode) const
{
    auto& state = ensureState(playerPrivate);

    // The Nexus playpump buffers a lot of data. Let's add it as if it had been buffered by the GstQueue2
    // (only when using in-memory buffering), so we get more realistic percentages.
    if (mode != GST_BUFFERING_STREAM || !(state.m_vidfilter || state.m_audfilter))
        return originalBufferingPercentage;

    // The Nexus playpump buffers a lot of data. Let's add it as if it had been buffered by the GstQueue2
    // (only when using in-memory buffering), so we get more realistic percentages.
    int correctedBufferingPercentage1 = originalBufferingPercentage;
    int correctedBufferingPercentage2 = originalBufferingPercentage;
    unsigned maxSizeBytes = 0;

    // We don't trust the buffering percentage when it's 0, better rely on current-level-bytes and compute a new buffer level accordingly.
    g_object_get(state.m_queue2.get(), "max-size-bytes", &maxSizeBytes, nullptr);
    if (!originalBufferingPercentage && state.m_queue2) {
        unsigned currentLevelBytes = 0;
        g_object_get(state.m_queue2.get(), "current-level-bytes", &currentLevelBytes, nullptr);
        correctedBufferingPercentage1 = currentLevelBytes > maxSizeBytes ? 100 : static_cast<int>(currentLevelBytes * 100 / maxSizeBytes);
    }

    unsigned playpumpBufferedBytes = 0;

    // We believe that the playpump stats are common for audio and video, so only asking the vidfilter or the audfilter
    // would be enough. Both of them expose the buffered_bytes property (yes, with an underscore!).
    GstElement* filter = state.m_vidfilter ? state.m_vidfilter.get() : state.m_audfilter.get();
    if (filter)
        g_object_get(GST_OBJECT(filter), "buffered_bytes", &playpumpBufferedBytes, nullptr);

    unsigned multiqueueBufferedBytes = 0;
    if (state.m_multiqueue) {
        GUniqueOutPtr<GstStructure> stats;
        g_object_get(state.m_multiqueue.get(), "stats", &stats.outPtr(), nullptr);
        for (const auto& queue : gstStructureGetArray<const GstStructure*>(stats.get(), "queues"_s))
            multiqueueBufferedBytes += gstStructureGet<unsigned>(queue, "bytes"_s).value_or(0);
    }

    // Current-level-bytes seems to be inacurate, so we compute its value from the buffering percentage.
    size_t currentLevelBytes = static_cast<size_t>(maxSizeBytes) * static_cast<size_t>(originalBufferingPercentage) / static_cast<size_t>(100)
        + static_cast<size_t>(playpumpBufferedBytes) + static_cast<size_t>(multiqueueBufferedBytes);
    correctedBufferingPercentage2 = currentLevelBytes > maxSizeBytes ? 100 : static_cast<int>(currentLevelBytes * 100 / maxSizeBytes);

    if (correctedBufferingPercentage2 >= 100)
        state.m_streamBufferingLevelMovingAverage.reset(100);
    int averagedBufferingPercentage = state.m_streamBufferingLevelMovingAverage.accumulate(correctedBufferingPercentage2);

    const char* extraElements = state.m_multiqueue ? "playpump and multiqueue" : "playpump";
    if (!originalBufferingPercentage) {
        GST_DEBUG("[Buffering] Buffering: mode: GST_BUFFERING_STREAM, status: %d%% (corrected to %d%% with current-level-bytes, "
            "to %d%% with %s content, and to %d%% with moving average).", originalBufferingPercentage, correctedBufferingPercentage1,
            correctedBufferingPercentage2, extraElements, averagedBufferingPercentage);
    } else {
        GST_DEBUG("[Buffering] Buffering: mode: GST_BUFFERING_STREAM, status: %d%% (corrected to %d%% with %s content and "
            "to %d%% with moving average).", originalBufferingPercentage, correctedBufferingPercentage2, extraElements,
            averagedBufferingPercentage);
    }

    return averagedBufferingPercentage;
}

void GStreamerQuirkBroadcomBase::resetBufferingPercentage(MediaPlayerPrivateGStreamer* playerPrivate, int bufferingPercentage) const
{
    auto& state = ensureState(playerPrivate);
    state.m_streamBufferingLevelMovingAverage.reset(bufferingPercentage);
}

void GStreamerQuirkBroadcomBase::setupBufferingPercentageCorrection(MediaPlayerPrivateGStreamer* playerPrivate, GstState currentState, GstState newState, GRefPtr<GstElement>&& element) const
{
    auto& state = ensureState(playerPrivate);

    // This code must support being run from different GStreamerQuirkBroadcomBase subclasses without breaking. Only the
    // first subclass instance should run.

    if (currentState == GST_STATE_NULL && newState == GST_STATE_READY) {
        bool alsoGetMultiqueue = false;
        if (!g_strcmp0(G_OBJECT_TYPE_NAME(element.get()), "GstBrcmVidFilter")) {
            state.m_vidfilter = element;
            alsoGetMultiqueue = true;
        } else if (!g_strcmp0(G_OBJECT_TYPE_NAME(element.get()), "GstBrcmAudFilter")) {
            state.m_audfilter = element;
            alsoGetMultiqueue = true;
        } else if (!g_strcmp0(G_OBJECT_TYPE_NAME(element.get()), "GstQueue2"))
            state.m_queue2 = element;

        // Might have been already retrieved by vidfilter or audfilter, whichever appeared first.
        if (alsoGetMultiqueue && !state.m_multiqueue) {
            // Also get the multiqueue (if there's one) attached to the vidfilter/aacparse+audfilter.
            // We'll need it later to correct the buffering level.
            for (auto* sinkPad : GstIteratorAdaptor<GstPad>(GUniquePtr<GstIterator>(gst_element_iterate_sink_pads(element.get())))) {
                GRefPtr<GstPad> peerSrcPad = adoptGRef(gst_pad_get_peer(sinkPad));
                if (!peerSrcPad)
                    continue; // And end the loop, because there's only one srcpad.
                GRefPtr<GstElement> peerElement = adoptGRef(GST_ELEMENT(gst_pad_get_parent(peerSrcPad.get())));

                // If it's NOT a multiqueue, it's probably a parser like aacparse. We try to traverse before it.
                if (peerElement && g_strcmp0(G_OBJECT_TYPE_NAME(element.get()), "GstMultiQueue")) {
                    for (auto* peerElementSinkPad : GstIteratorAdaptor<GstPad>(GUniquePtr<GstIterator>(gst_element_iterate_sink_pads(peerElement.get())))) {
                        peerSrcPad = adoptGRef(gst_pad_get_peer(peerElementSinkPad));
                        if (!peerSrcPad)
                            continue; // And end the loop.
                        // Now we hopefully have peerElement pointing to the multiqueue.
                        peerElement = adoptGRef(GST_ELEMENT(gst_pad_get_parent(peerSrcPad.get())));
                        break;
                    }
                }

                // The multiqueue reference is useless if we can't access its stats (on older GStreamer versions).
                if (peerElement && !g_strcmp0(G_OBJECT_TYPE_NAME(element.get()), "GstMultiQueue")
                    && gstObjectHasProperty(peerElement.get(), "stats"))
                    state.m_multiqueue = peerElement;
                break;
            }
        }
    } else if (currentState == GST_STATE_READY && newState == GST_STATE_NULL) {
        if (!g_strcmp0(G_OBJECT_TYPE_NAME(element.get()), "GstBrcmVidFilter"))
            state.m_vidfilter = nullptr;
        else if (!g_strcmp0(G_OBJECT_TYPE_NAME(element.get()), "GstBrcmAudFilter"))
            state.m_audfilter = nullptr;
        else if (!g_strcmp0(G_OBJECT_TYPE_NAME(element.get()), "GstMultiQueue") && element == state.m_multiqueue.get())
            state.m_multiqueue = nullptr;
        else if (!g_strcmp0(G_OBJECT_TYPE_NAME(element.get()), "GstQueue2"))
            state.m_queue2 = nullptr;
    }
}

GStreamerQuirkBroadcomBase::GStreamerQuirkBroadcomBaseState& GStreamerQuirkBroadcomBase::ensureState(MediaPlayerPrivateGStreamer* playerPrivate) const
{
    GStreamerQuirkBase::GStreamerQuirkState* state = playerPrivate->quirkState(this);
    if (!state) {
        GST_DEBUG("%s %p setting up quirk state on MediaPlayerPrivate %p", identifier().characters(), this, playerPrivate);
        playerPrivate->setQuirkState(this, makeUnique<GStreamerQuirkBroadcomBaseState>());
        state = playerPrivate->quirkState(this);
    }
    return reinterpret_cast<GStreamerQuirkBroadcomBaseState&>(*state);
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER)
