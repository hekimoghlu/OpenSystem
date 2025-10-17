/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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

#include "Timer.h"
#include <wtf/HashMap.h>
#include <wtf/IteratorRange.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS AVAssetTrack;
OBJC_CLASS AVPlayerItem;
OBJC_CLASS AVMediaSelectionGroup;
OBJC_CLASS AVMediaSelectionOption;

namespace WebCore {

class MediaSelectionGroupAVFObjC;

class MediaSelectionOptionAVFObjC : public RefCounted<MediaSelectionOptionAVFObjC> {
public:
    static Ref<MediaSelectionOptionAVFObjC> create(MediaSelectionGroupAVFObjC&, AVMediaSelectionOption *);

    void setSelected(bool);
    bool selected() const;

    int index() const;

    AVMediaSelectionOption *avMediaSelectionOption() const { return m_mediaSelectionOption.get(); }
    AVAssetTrack *assetTrack() const;

private:
    friend class MediaSelectionGroupAVFObjC;
    MediaSelectionOptionAVFObjC(MediaSelectionGroupAVFObjC&, AVMediaSelectionOption *);

    void clearGroup() { m_group = nullptr; }

    MediaSelectionGroupAVFObjC* m_group;
    RetainPtr<AVMediaSelectionOption> m_mediaSelectionOption;
};

class MediaSelectionGroupAVFObjC : public RefCounted<MediaSelectionGroupAVFObjC> {
public:
    static Ref<MediaSelectionGroupAVFObjC> create(AVPlayerItem*, AVMediaSelectionGroup*, const Vector<String>& characteristics);
    ~MediaSelectionGroupAVFObjC();

    void setSelectedOption(MediaSelectionOptionAVFObjC*);
    MediaSelectionOptionAVFObjC* selectedOption() const { return m_selectedOption; }

    void updateOptions(const Vector<String>& characteristics);

    using OptionContainer = UncheckedKeyHashMap<CFTypeRef, RefPtr<MediaSelectionOptionAVFObjC>>;
    typename OptionContainer::ValuesIteratorRange options() { return m_options.values(); }

    AVMediaSelectionGroup *avMediaSelectionGroup() const { return m_mediaSelectionGroup.get(); }

private:
    MediaSelectionGroupAVFObjC(AVPlayerItem*, AVMediaSelectionGroup*, const Vector<String>& characteristics);

    void selectionTimerFired();

    RetainPtr<AVPlayerItem> m_playerItem;
    RetainPtr<AVMediaSelectionGroup> m_mediaSelectionGroup;
    OptionContainer m_options;
    MediaSelectionOptionAVFObjC* m_selectedOption { nullptr };
    Timer m_selectionTimer;
    bool m_shouldSelectOptionAutomatically { true };
};

}

#endif // ENABLE(VIDEO)
