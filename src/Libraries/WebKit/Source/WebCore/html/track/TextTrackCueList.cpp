/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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

#if ENABLE(VIDEO)

#include "TextTrackCueList.h"
#include <algorithm>
#include <ranges>

// Checking sorting is too slow for general use; turn it on explicitly when working on this class.
#undef CHECK_SORTING

#ifdef CHECK_SORTING
#define ASSERT_SORTED(range) ASSERT(std::ranges::is_sorted(range, cueSortsBefore))
#else
#define ASSERT_SORTED(range) ((void)0)
#endif

namespace WebCore {

static inline bool cueSortsBefore(const RefPtr<TextTrackCue>& a, const RefPtr<TextTrackCue>& b)
{
    if (a->startMediaTime() < b->startMediaTime())
        return true;

    return a->startMediaTime() == b->startMediaTime() && a->endMediaTime() > b->endMediaTime();
}

Ref<TextTrackCueList> TextTrackCueList::create()
{
    return adoptRef(*new TextTrackCueList);
}

void TextTrackCueList::didMoveToNewDocument(Document& newDocument)
{
    for (RefPtr cue : m_vector)
        cue->didMoveToNewDocument(newDocument);
}

unsigned TextTrackCueList::cueIndex(const TextTrackCue& cue) const
{
    ASSERT(m_vector.contains(&cue));
    return m_vector.find(&cue);
}

TextTrackCue* TextTrackCueList::item(unsigned index) const
{
    if (index >= m_vector.size())
        return nullptr;
    return m_vector[index].get();
}

TextTrackCue* TextTrackCueList::getCueById(const String& id) const
{
    for (auto& cue : m_vector) {
        if (cue->id() == id)
            return cue.get();
    }
    return nullptr;
}

TextTrackCueList& TextTrackCueList::activeCues()
{
    if (!m_activeCues)
        m_activeCues = create();

    Vector<RefPtr<TextTrackCue>> activeCuesVector;
    for (auto& cue : m_vector) {
        if (cue->isActive())
            activeCuesVector.append(cue);
    }
    ASSERT_SORTED(activeCuesVector);
    m_activeCues->m_vector = WTFMove(activeCuesVector);

    // FIXME: This list of active cues is not updated as cues are added, removed, become active, and become inactive.
    // Instead it is only updated each time this function is called again. That is not consistent with other dynamic DOM lists.
    return *m_activeCues;
}

void TextTrackCueList::add(Ref<TextTrackCue>&& cue)
{
    ASSERT(!m_vector.contains(cue.ptr()));

    RefPtr<TextTrackCue> cueRefPtr { WTFMove(cue) };
    unsigned insertionPosition = std::ranges::upper_bound(m_vector, cueRefPtr, cueSortsBefore) - m_vector.begin();
    ASSERT_SORTED(m_vector);
    m_vector.insert(insertionPosition, WTFMove(cueRefPtr));
    ASSERT_SORTED(m_vector);
}

void TextTrackCueList::remove(TextTrackCue& cue)
{
    ASSERT_SORTED(m_vector);
    m_vector.remove(cueIndex(cue));
    ASSERT_SORTED(m_vector);
}

void TextTrackCueList::clear()
{
    m_vector.clear();
    if (m_activeCues)
        m_activeCues->m_vector.clear();
}

void TextTrackCueList::updateCueIndex(const TextTrackCue& cue)
{
    auto vectorSpan = m_vector.mutableSpan();
    auto cueIndex = this->cueIndex(cue);
    auto valuesUntilCue = vectorSpan.first(cueIndex);
    auto cuePosition = vectorSpan.subspan(cueIndex).begin();
    auto valuesAfterCue = vectorSpan.subspan(cueIndex + 1);
    ASSERT_SORTED(valuesUntilCue);
    ASSERT_SORTED(valuesAfterCue);

    auto reinsertionPosition = std::ranges::upper_bound(valuesUntilCue, *cuePosition, cueSortsBefore);
    if (std::to_address(reinsertionPosition) != std::to_address(cuePosition))
        std::rotate(reinsertionPosition, cuePosition, valuesAfterCue.begin());
    else {
        reinsertionPosition = std::ranges::upper_bound(valuesAfterCue, *cuePosition, cueSortsBefore);
        if (std::to_address(reinsertionPosition) != valuesAfterCue.data())
            std::rotate(cuePosition, valuesAfterCue.begin(), reinsertionPosition);
    }

    ASSERT_SORTED(m_vector);
}

} // namespace WebCore

#endif
