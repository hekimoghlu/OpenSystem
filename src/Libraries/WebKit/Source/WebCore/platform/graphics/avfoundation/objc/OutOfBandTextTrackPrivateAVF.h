/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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
#ifndef OutOfBandTextTrackPrivateAVF_h
#define OutOfBandTextTrackPrivateAVF_h

#if ENABLE(VIDEO) && (USE(AVFOUNDATION) || PLATFORM(IOS_FAMILY))

#include "InbandTextTrackPrivateAVF.h"
#include <wtf/TZoneMallocInlines.h>

OBJC_CLASS AVMediaSelectionOption;

namespace WebCore {
    
class OutOfBandTextTrackPrivateAVF : public InbandTextTrackPrivateAVF {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(OutOfBandTextTrackPrivateAVF);
public:
    static Ref<OutOfBandTextTrackPrivateAVF> create(AVFInbandTrackParent* player,  AVMediaSelectionOption* selection, TrackID trackID)
    {
        return adoptRef(*new OutOfBandTextTrackPrivateAVF(player, selection, trackID));
    }
    
    void processCue(CFArrayRef, CFArrayRef, const MediaTime&) override { }
    void resetCueValues() override { }
    
    Category textTrackCategory() const override { return OutOfBand; }
    
    AVMediaSelectionOption* mediaSelectionOption() const { return m_mediaSelectionOption.get(); }
    
protected:
    OutOfBandTextTrackPrivateAVF(AVFInbandTrackParent* player, AVMediaSelectionOption* selection, TrackID trackID)
        : InbandTextTrackPrivateAVF(player, trackID, InbandTextTrackPrivate::CueFormat::Generic)
        , m_mediaSelectionOption(selection)
    {
    }
    
    RetainPtr<AVMediaSelectionOption> m_mediaSelectionOption;
};
    
}

#endif

#endif // OutOfBandTextTrackPrivateAVF_h
