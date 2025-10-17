/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "TextTrackPrivateRemoteConfiguration.h"
#include <WebCore/InbandTextTrackPrivate.h>
#include <WebCore/MediaPlayerIdentifier.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class InbandGenericCue;
class ISOWebVTTCue;
}

namespace WebKit {

class GPUProcessConnection;
class MediaPlayerPrivateRemote;

class TextTrackPrivateRemote final : public WebCore::InbandTextTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED(TextTrackPrivateRemote);
    WTF_MAKE_NONCOPYABLE(TextTrackPrivateRemote)
public:

    static Ref<TextTrackPrivateRemote> create(GPUProcessConnection& gpuProcessConnection, WebCore::MediaPlayerIdentifier playerIdentifier, TextTrackPrivateRemoteConfiguration&& configuration)
    {
        return adoptRef(*new TextTrackPrivateRemote(gpuProcessConnection, playerIdentifier, WTFMove(configuration)));
    }

    void addDataCue(MediaTime&& start, MediaTime&& end, std::span<const uint8_t>);

#if ENABLE(DATACUE_VALUE)
    using SerializedPlatformDataCueValue = WebCore::SerializedPlatformDataCueValue;
    void addDataCueWithType(MediaTime&& start, MediaTime&& end, SerializedPlatformDataCueValue&&, String&&);
    void updateDataCue(MediaTime&& start, MediaTime&& end, SerializedPlatformDataCueValue&&);
    void removeDataCue(MediaTime&& start, MediaTime&& end, SerializedPlatformDataCueValue&&);
#endif

    using InbandGenericCue = WebCore::InbandGenericCue;
    void addGenericCue(Ref<InbandGenericCue>);
    void updateGenericCue(Ref<InbandGenericCue>);
    void removeGenericCue(Ref<InbandGenericCue>);

    using ISOWebVTTCue = WebCore::ISOWebVTTCue;
    void parseWebVTTFileHeader(String&&);
    void parseWebVTTCueData(std::span<const uint8_t>);
    void parseWebVTTCueDataStruct(ISOWebVTTCue&&);

    void updateConfiguration(TextTrackPrivateRemoteConfiguration&&);

    WebCore::TrackID id() const final { return m_id; }
    AtomString label() const final { return AtomString { m_label }; }
    AtomString language() const final { return AtomString { m_language }; }
    int trackIndex() const final { return m_trackIndex; }
    AtomString inBandMetadataTrackDispatchType() const final { return AtomString { m_inBandMetadataTrackDispatchType }; }

    using TextTrackKind = WebCore::InbandTextTrackPrivate::Kind;
    TextTrackKind kind() const final { return m_kind; }

    using TextTrackMode = WebCore::InbandTextTrackPrivate::Mode;
    void setMode(TextTrackMode) final;

    bool isClosedCaptions() const final { return m_isClosedCaptions; }
    bool isSDH() const final { return m_isSDH; }
    bool containsOnlyForcedSubtitles() const final { return m_containsOnlyForcedSubtitles; }
    bool isMainProgramContent() const final { return m_isMainProgramContent; }
    bool isEasyToRead() const final { return m_isEasyToRead; }
    bool isDefault() const final { return m_isDefault; }
    MediaTime startTimeVariance() const final { return m_startTimeVariance; }

private:
    TextTrackPrivateRemote(GPUProcessConnection&, WebCore::MediaPlayerIdentifier, TextTrackPrivateRemoteConfiguration&&);

    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    String m_label;
    String m_language;
    int m_trackIndex { -1 };
    String m_inBandMetadataTrackDispatchType;
    MediaTime m_startTimeVariance { MediaTime::zeroTime() };
    WebCore::TrackID m_id;
    WebCore::MediaPlayerIdentifier m_playerIdentifier;

    TextTrackKind m_kind { TextTrackKind::None };
    bool m_isClosedCaptions { false };
    bool m_isSDH { false };
    bool m_containsOnlyForcedSubtitles { false };
    bool m_isMainProgramContent { true };
    bool m_isEasyToRead { false };
    bool m_isDefault { false };
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
