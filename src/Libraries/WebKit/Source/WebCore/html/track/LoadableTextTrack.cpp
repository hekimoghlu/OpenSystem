/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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
#include "config.h"
#include "LoadableTextTrack.h"

#if ENABLE(VIDEO)

#include "ElementInlines.h"
#include "ScriptExecutionContext.h"
#include "TextTrackCueList.h"
#include "VTTCue.h"
#include "VTTRegionList.h"
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LoadableTextTrack);

LoadableTextTrack::LoadableTextTrack(HTMLTrackElement& track, const AtomString& kind, const AtomString& label, const AtomString& language)
    : TextTrack(track.scriptExecutionContext(), kind, emptyAtom(), label, language, TrackElement)
    , m_trackElement(track)
{
}

Ref<LoadableTextTrack> LoadableTextTrack::create(HTMLTrackElement& track, const AtomString& kind, const AtomString& label, const AtomString& language)
{
    auto textTrack = adoptRef(*new LoadableTextTrack(track, kind, label, language));
    textTrack->suspendIfNeeded();
    return textTrack;
}

void LoadableTextTrack::scheduleLoad(const URL& url)
{
    ASSERT(!url.isEmpty());

    if (url == m_url)
        return;

    // When src attribute is changed we need to flush all collected track data
    removeAllCues();

    if (!m_trackElement)
        return;

    // 4.8.10.12.3 Sourcing out-of-band text tracks (continued)

    // 2. Let URL be the track URL of the track element.
    m_url = url;
    
    if (m_loadPending)
        return;
    
    // 3. Asynchronously run the remaining steps, while continuing with whatever task
    // was responsible for creating the text track or changing the text track mode.
    m_trackElement->scheduleTask([this]() mutable {
        SetForScope loadPending { m_loadPending, true, false };

        if (m_loader)
            m_loader->cancelLoad();

        if (!m_trackElement)
            return;

        // 4.8.10.12.3 Sourcing out-of-band text tracks (continued)

        // 4. Download: If URL is not the empty string, perform a potentially CORS-enabled fetch of URL, with the
        // mode being the state of the media element's crossorigin content attribute, the origin being the
        // origin of the media element's Document, and the default origin behaviour set to fail.
        m_loader = makeUnique<TextTrackLoader>(static_cast<TextTrackLoaderClient&>(*this), m_trackElement->document());
        if (!m_loader->load(m_url, *m_trackElement))
            m_trackElement->didCompleteLoad(HTMLTrackElement::Failure);
    });
}

void LoadableTextTrack::newCuesAvailable(TextTrackLoader& loader)
{
    ASSERT_UNUSED(loader, m_loader.get() == &loader);

    if (!m_cues)
        m_cues = TextTrackCueList::create();    

    for (auto& newCue : m_loader->getNewCues()) {
        newCue->setTrack(this);
        INFO_LOG(LOGIDENTIFIER, newCue.get());
        m_cues->add(WTFMove(newCue));
    }

    TextTrack::newCuesAvailable(*m_cues);
}

void LoadableTextTrack::cueLoadingCompleted(TextTrackLoader& loader, bool loadingFailed)
{
    ASSERT_UNUSED(loader, m_loader.get() == &loader);

    if (!m_trackElement)
        return;

    INFO_LOG(LOGIDENTIFIER);

    m_trackElement->didCompleteLoad(loadingFailed ? HTMLTrackElement::Failure : HTMLTrackElement::Success);
}

void LoadableTextTrack::newRegionsAvailable(TextTrackLoader& loader)
{
    ASSERT_UNUSED(loader, m_loader.get() == &loader);
    for (auto& newRegion : m_loader->getNewRegions())
        regions()->add(WTFMove(newRegion));
}

void LoadableTextTrack::newStyleSheetsAvailable(TextTrackLoader& loader)
{
    ASSERT_UNUSED(loader, m_loader.get() == &loader);
    m_styleSheets = m_loader->getNewStyleSheets();
}

AtomString LoadableTextTrack::id() const
{
    if (!m_trackElement)
        return emptyAtom();
    return m_trackElement->attributeWithoutSynchronization(idAttr);
}

size_t LoadableTextTrack::trackElementIndex()
{
    ASSERT(m_trackElement);
    ASSERT(m_trackElement->parentNode());

    size_t index = 0;
    for (RefPtr<Node> node = m_trackElement->parentNode()->firstChild(); node; node = node->nextSibling()) {
        if (!node->hasTagName(trackTag) || !node->parentNode())
            continue;
        if (node.get() == m_trackElement.get())
            return index;
        ++index;
    }
    ASSERT_NOT_REACHED();

    return 0;
}

bool LoadableTextTrack::isDefault() const
{
    return m_trackElement && m_trackElement->hasAttributeWithoutSynchronization(defaultAttr);
}

} // namespace WebCore

#endif
