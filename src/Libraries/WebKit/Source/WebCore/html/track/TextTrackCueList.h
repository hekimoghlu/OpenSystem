/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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

#include "TextTrackCue.h"

namespace WebCore {

class Document;

class TextTrackCueList : public RefCounted<TextTrackCueList> {
public:
    static Ref<TextTrackCueList> create();

    void didMoveToNewDocument(Document&);

    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    unsigned length() const;
    TextTrackCue* item(unsigned index) const;
    TextTrackCue* getCueById(const String&) const;

    unsigned cueIndex(const TextTrackCue&) const;

    void add(Ref<TextTrackCue>&&);
    void remove(TextTrackCue&);
    void updateCueIndex(const TextTrackCue&);

    void clear();

    TextTrackCueList& activeCues();

private:
    TextTrackCueList() = default;

    Vector<RefPtr<TextTrackCue>> m_vector;
    RefPtr<TextTrackCueList> m_activeCues;
};

inline unsigned TextTrackCueList::length() const
{
    return m_vector.size();
}

} // namespace WebCore

#endif // ENABLE(VIDEO)
