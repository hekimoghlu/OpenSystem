/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
#ifndef InbandTextTrackPrivateAVF_h
#define InbandTextTrackPrivateAVF_h

#if ENABLE(VIDEO) && (USE(AVFOUNDATION) || PLATFORM(IOS_FAMILY))

#include "InbandTextTrackPrivate.h"
#include "InbandTextTrackPrivateClient.h"
#include <wtf/TZoneMalloc.h>

typedef const struct opaqueCMFormatDescription* CMFormatDescriptionRef;

namespace JSC {
class ArrayBuffer;
}

namespace WebCore {

class AVFInbandTrackParent {
public:
    virtual ~AVFInbandTrackParent();
    
    virtual void trackModeChanged() = 0;
};

class InbandTextTrackPrivateAVF : public InbandTextTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED(InbandTextTrackPrivateAVF);
public:
    virtual ~InbandTextTrackPrivateAVF();

    TrackID id() const final { return m_trackID; }

    void setMode(InbandTextTrackPrivate::Mode) override;

    int trackIndex() const override { return m_index; }
    void setTextTrackIndex(int index) { m_index = index; }

    virtual void disconnect();

    bool hasBeenReported() const { return m_hasBeenReported; }
    void setHasBeenReported(bool reported) { m_hasBeenReported = reported; }

    virtual void processCue(CFArrayRef attributedStrings, CFArrayRef nativeSamples, const MediaTime&);
    virtual void resetCueValues();

    void beginSeeking();
    void endSeeking() { m_seeking = false; }
    bool seeking() const { return m_seeking; }
    
    enum Category {
        LegacyClosedCaption,
        OutOfBand,
        InBand
    };
    virtual Category textTrackCategory() const = 0;
    
    MediaTime startTimeVariance() const override { return MediaTime(1, 4); }

    virtual bool readNativeSampleBuffer(CFArrayRef nativeSamples, CFIndex, RefPtr<JSC::ArrayBuffer>&, MediaTime&, CMFormatDescriptionRef&);
    
protected:
    InbandTextTrackPrivateAVF(AVFInbandTrackParent*, TrackID, CueFormat);

    Ref<InbandGenericCue> processCueAttributes(CFAttributedStringRef);
    void processAttributedStrings(CFArrayRef, const MediaTime&);
    void processNativeSamples(CFArrayRef, const MediaTime&);
    void removeCompletedCues();

    Vector<uint8_t> m_sampleInputBuffer;

private:
#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "InbandTextTrackPrivateAVF"_s; }
#endif

    MediaTime m_currentCueStartTime;
    MediaTime m_currentCueEndTime;

    Vector<Ref<InbandGenericCue>> m_cues;
    AVFInbandTrackParent* m_owner;

    enum PendingCueStatus {
        None,
        DeliveredDuringSeek,
        Valid
    };
    PendingCueStatus m_pendingCueStatus;

    int m_index;
    bool m_hasBeenReported;
    bool m_seeking;
    bool m_haveReportedVTTHeader;
    const TrackID m_trackID { 0 };
};

} // namespace WebCore

#endif //  ENABLE(VIDEO) && (USE(AVFOUNDATION) || PLATFORM(IOS_FAMILY))

#endif // InbandTextTrackPrivateAVF_h
