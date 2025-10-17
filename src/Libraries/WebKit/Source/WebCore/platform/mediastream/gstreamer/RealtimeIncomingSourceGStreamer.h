/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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

#if USE(GSTREAMER_WEBRTC)

#include "GRefPtrGStreamer.h"
#include "RealtimeMediaSource.h"

namespace WebCore {

class RealtimeIncomingSourceGStreamer : public RealtimeMediaSource, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RealtimeIncomingSourceGStreamer> {
public:
    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

    GstElement* bin() const { return m_bin.get(); }
    bool setBin(const GRefPtr<GstElement>&);

    bool hasClient(const GRefPtr<GstElement>&);
    int registerClient(GRefPtr<GstElement>&&);
    void unregisterClient(int);

    void handleUpstreamEvent(GRefPtr<GstEvent>&&);
    bool handleUpstreamQuery(GstQuery*);
    void handleDownstreamEvent(GstElement* sink, GRefPtr<GstEvent>&&);

    void tearDown();

protected:
    RealtimeIncomingSourceGStreamer(const CaptureDevice&);

private:
    // RealtimeMediaSource API
    const RealtimeMediaSourceCapabilities& capabilities() final;

    virtual void dispatchSample(GRefPtr<GstSample>&&) = 0;

    void forEachClient(Function<void(GstElement*)>&&);

    GRefPtr<GstElement> m_bin;
    GRefPtr<GstElement> m_sink;
    Lock m_clientLock;
    HashMap<int, GRefPtr<GstElement>> m_clients WTF_GUARDED_BY_LOCK(m_clientLock);
};

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
