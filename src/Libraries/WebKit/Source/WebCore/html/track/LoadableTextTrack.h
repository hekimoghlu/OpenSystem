/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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

#include "HTMLTrackElement.h"
#include "TextTrack.h"
#include "TextTrackLoader.h"

namespace WebCore {

class LoadableTextTrack final : public TextTrack, private TextTrackLoaderClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LoadableTextTrack);
public:
    static Ref<LoadableTextTrack> create(HTMLTrackElement&, const AtomString& kind, const AtomString& label, const AtomString& language);

    void scheduleLoad(const URL&);

    size_t trackElementIndex();
    HTMLTrackElement* trackElement() const { return m_trackElement.get(); }

private:
    LoadableTextTrack(HTMLTrackElement&, const AtomString& kind, const AtomString& label, const AtomString& language);

    void newCuesAvailable(TextTrackLoader&) final;
    void cueLoadingCompleted(TextTrackLoader&, bool loadingFailed) final;
    void newRegionsAvailable(TextTrackLoader&) final;
    void newStyleSheetsAvailable(TextTrackLoader&) final;

    AtomString id() const final;
    bool isDefault() const final;

    void loadTimerFired();

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const override { return "LoadableTextTrack"_s; }
#endif

    WeakPtr<HTMLTrackElement, WeakPtrImplWithEventTargetData> m_trackElement;
    std::unique_ptr<TextTrackLoader> m_loader;
    URL m_url;
    bool m_loadPending { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::LoadableTextTrack)
    static bool isType(const WebCore::TextTrack& track) { return track.trackType() == WebCore::TextTrack::TrackElement; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(VIDEO)
