/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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

#if ENABLE(VIDEO) && USE(AVFOUNDATION)

#include "InbandTextTrackPrivateAVF.h"
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS AVAsset;
OBJC_CLASS AVMediaSelectionGroup;
OBJC_CLASS AVMediaSelectionOption;

namespace WebCore {

class InbandTextTrackPrivateAVFObjC : public InbandTextTrackPrivateAVF {
    WTF_MAKE_TZONE_ALLOCATED(InbandTextTrackPrivateAVFObjC);
public:
    static Ref<InbandTextTrackPrivateAVFObjC> create(AVFInbandTrackParent* player,  AVMediaSelectionGroup *group, AVMediaSelectionOption *selection, TrackID trackID, InbandTextTrackPrivate::CueFormat format)
    {
        return adoptRef(*new InbandTextTrackPrivateAVFObjC(player, group, selection, trackID, format));
    }

    ~InbandTextTrackPrivateAVFObjC() = default;

    InbandTextTrackPrivate::Kind kind() const override;
    bool isClosedCaptions() const override;
    bool isSDH() const override;
    bool containsOnlyForcedSubtitles() const override;
    bool isMainProgramContent() const override;
    bool isEasyToRead() const override;
    AtomString label() const override;
    AtomString language() const override;
    bool isDefault() const override;

    void disconnect() override;

    Category textTrackCategory() const override { return InBand; }
    
    AVMediaSelectionOption *mediaSelectionOption() const { return m_mediaSelectionOption.get(); }

protected:
    InbandTextTrackPrivateAVFObjC(AVFInbandTrackParent*, AVMediaSelectionGroup *, AVMediaSelectionOption *, TrackID, InbandTextTrackPrivate::CueFormat);
    
    RetainPtr<AVMediaSelectionGroup> m_mediaSelectionGroup;
    RetainPtr<AVMediaSelectionOption> m_mediaSelectionOption;
};

}

#endif
