/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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
#ifndef InbandMetadataTextTrackPrivateAVF_h
#define InbandMetadataTextTrackPrivateAVF_h

#if ENABLE(VIDEO) && USE(AVFOUNDATION)
#include "InbandTextTrackPrivate.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

#if ENABLE(DATACUE_VALUE)
struct IncompleteMetaDataCue {
    RefPtr<SerializedPlatformDataCue> cueData;
    MediaTime startTime;
};
#endif

class InbandMetadataTextTrackPrivateAVF : public InbandTextTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED(InbandMetadataTextTrackPrivateAVF);
public:
    static Ref<InbandMetadataTextTrackPrivateAVF> create(Kind, TrackID, CueFormat);

    ~InbandMetadataTextTrackPrivateAVF();

    Kind kind() const override { return m_kind; }
    TrackID id() const override { return m_id; }
    AtomString inBandMetadataTrackDispatchType() const override { return m_inBandMetadataTrackDispatchType; }
    void setInBandMetadataTrackDispatchType(const AtomString& value) { m_inBandMetadataTrackDispatchType = value; }

#if ENABLE(DATACUE_VALUE)
    void addDataCue(const MediaTime& start, const MediaTime& end, Ref<SerializedPlatformDataCue>&&, const String&);
    void updatePendingCueEndTimes(const MediaTime&);
#endif

    void flushPartialCues();

private:
    InbandMetadataTextTrackPrivateAVF(Kind, TrackID, CueFormat);

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "InbandMetadataTextTrackPrivateAVF"_s; }
#endif

    Kind m_kind;
    TrackID m_id;
    AtomString m_inBandMetadataTrackDispatchType;
    MediaTime m_currentCueStartTime;
#if ENABLE(DATACUE_VALUE)
    Vector<IncompleteMetaDataCue> m_incompleteCues;
#endif
};

} // namespace WebCore

#endif // ENABLE(VIDEO) && USE(AVFOUNDATION)

#endif // InbandMetadataTextTrackPrivateAVF_h
