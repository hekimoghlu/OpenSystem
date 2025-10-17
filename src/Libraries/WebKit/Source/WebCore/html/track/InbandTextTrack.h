/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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

#if ENABLE(VIDEO)

#include "InbandTextTrackPrivateClient.h"
#include "TextTrack.h"

namespace WebCore {

class InbandTextTrack : public TextTrack, private InbandTextTrackPrivateClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(InbandTextTrack);
public:
    static Ref<InbandTextTrack> create(ScriptExecutionContext&, InbandTextTrackPrivate&);
    virtual ~InbandTextTrack();

    bool isClosedCaptions() const override;
    bool isSDH() const override;
    bool containsOnlyForcedSubtitles() const override;
    bool isMainProgramContent() const override;
    bool isEasyToRead() const override;
    void setMode(Mode) override;
    bool isDefault() const override;
    size_t inbandTrackIndex();

    AtomString inBandMetadataTrackDispatchType() const override;

    void setPrivate(InbandTextTrackPrivate&);
#if !RELEASE_LOG_DISABLED
    void setLogger(const Logger&, uint64_t) final;
#endif

protected:
    InbandTextTrack(ScriptExecutionContext&, InbandTextTrackPrivate&);

    void setModeInternal(Mode);
    void updateKindFromPrivate();

    Ref<InbandTextTrackPrivate> m_private;

    MediaTime startTimeVariance() const override;

private:
    bool isInband() const final { return true; }
    void idChanged(TrackID) override;
    void labelChanged(const AtomString&) override;
    void languageChanged(const AtomString&) override;
    void willRemove() override;

    void addDataCue(const MediaTime&, const MediaTime&, std::span<const uint8_t>) override { ASSERT_NOT_REACHED(); }

#if ENABLE(DATACUE_VALUE)
    void addDataCue(const MediaTime&, const MediaTime&, Ref<SerializedPlatformDataCue>&&, const String&) override { ASSERT_NOT_REACHED(); }
    void updateDataCue(const MediaTime&, const MediaTime&, SerializedPlatformDataCue&) override { ASSERT_NOT_REACHED(); }
    void removeDataCue(const MediaTime&, const MediaTime&, SerializedPlatformDataCue&) override { ASSERT_NOT_REACHED(); }
#endif

    void addGenericCue(InbandGenericCue&) override { ASSERT_NOT_REACHED(); }
    void updateGenericCue(InbandGenericCue&) override { ASSERT_NOT_REACHED(); }
    void removeGenericCue(InbandGenericCue&) override { ASSERT_NOT_REACHED(); }

    void parseWebVTTFileHeader(String&&) override { ASSERT_NOT_REACHED(); }
    void parseWebVTTCueData(std::span<const uint8_t>) override { ASSERT_NOT_REACHED(); }
    void parseWebVTTCueData(ISOWebVTTCue&&) override { ASSERT_NOT_REACHED(); }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::InbandTextTrack)
    static bool isType(const WebCore::TextTrack& track) { return track.isInband(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(VIDEO)
