/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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

#include "GUniquePtrGStreamer.h"
#include "RealtimeIncomingSourceGStreamer.h"

namespace WebCore {

class RealtimeIncomingVideoSourceGStreamer : public RealtimeIncomingSourceGStreamer {
public:
    static Ref<RealtimeIncomingVideoSourceGStreamer> create(AtomString&& videoTrackId) { return adoptRef(*new RealtimeIncomingVideoSourceGStreamer(WTFMove(videoTrackId))); }
    ~RealtimeIncomingVideoSourceGStreamer() = default;

    // RealtimeMediaSource API
    bool isIncomingVideoSource() const final { return true; }

    const GstStructure* stats();

protected:
    RealtimeIncomingVideoSourceGStreamer(AtomString&&);

private:
    // RealtimeMediaSource API
    void settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag>) final;
    const RealtimeMediaSourceSettings& settings() final;

    // RealtimeIncomingSourceGStreamer API
    void dispatchSample(GRefPtr<GstSample>&&) final;

    void ensureSizeAndFramerate(const GRefPtr<GstCaps>&);

    std::optional<RealtimeMediaSourceSettings> m_currentSettings;
    GUniquePtr<GstStructure> m_stats;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::RealtimeIncomingVideoSourceGStreamer)
    static bool isType(const WebCore::RealtimeMediaSource& source) { return source.isIncomingVideoSource(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(GSTREAMER_WEBRTC)
