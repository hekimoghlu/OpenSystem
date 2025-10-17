/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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

#if ENABLE(VIDEO) && (USE(AVFOUNDATION) || PLATFORM(IOS_FAMILY))

#include "InbandTextTrackPrivate.h"
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS AVTimedMetadataGroup;
OBJC_CLASS NSLocale;

namespace WebCore {

class InbandChapterTrackPrivateAVFObjC : public InbandTextTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED(InbandChapterTrackPrivateAVFObjC);
public:
    static Ref<InbandChapterTrackPrivateAVFObjC> create(RetainPtr<NSLocale> locale, TrackID trackID)
    {
        return adoptRef(*new InbandChapterTrackPrivateAVFObjC(WTFMove(locale), trackID));
    }

    virtual ~InbandChapterTrackPrivateAVFObjC() = default;

    TrackID id() const final { return m_id; }
    InbandTextTrackPrivate::Kind kind() const final { return InbandTextTrackPrivate::Kind::Chapters; }
    AtomString language() const final;

    int trackIndex() const final { return m_index; }
    void setTextTrackIndex(int index) { m_index = index; }

    void processChapters(RetainPtr<NSArray<AVTimedMetadataGroup *>>);

private:
    InbandChapterTrackPrivateAVFObjC(RetainPtr<NSLocale>, TrackID);

    AtomString inBandMetadataTrackDispatchType() const final { return "com.apple.chapters"_s; }
    ASCIILiteral logClassName() const final { return "InbandChapterTrackPrivateAVFObjC"_s; }

    struct ChapterData {
        MediaTime m_startTime;
        MediaTime m_duration;
        String m_title;

        friend bool operator==(const ChapterData&, const ChapterData&) = default;
    };

    Vector<ChapterData> m_processedChapters;
    RetainPtr<NSLocale> m_locale;
    mutable AtomString m_language;
    const TrackID m_id;
    int m_index { 0 };
};

} // namespace WebCore

#endif //  ENABLE(VIDEO) && (USE(AVFOUNDATION) || PLATFORM(IOS_FAMILY))
