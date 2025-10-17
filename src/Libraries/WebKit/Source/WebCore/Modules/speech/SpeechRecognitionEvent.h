/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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

#include "Event.h"

namespace WebCore {

class SpeechRecognitionResultList;

class SpeechRecognitionEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SpeechRecognitionEvent);
public:
    struct Init : EventInit {
        uint64_t resultIndex;
        RefPtr<SpeechRecognitionResultList> results;
    };

    static Ref<SpeechRecognitionEvent> create(const AtomString&, Init&&, IsTrusted = IsTrusted::No);
    static Ref<SpeechRecognitionEvent> create(const AtomString&, uint64_t resultIndex, RefPtr<SpeechRecognitionResultList>&&);

    virtual ~SpeechRecognitionEvent();

    uint64_t resultIndex() const { return m_resultIndex; }
    SpeechRecognitionResultList* results() const { return m_results.get(); }

private:
    SpeechRecognitionEvent(const AtomString&, Init&&, IsTrusted);
    SpeechRecognitionEvent(const AtomString&, uint64_t resultIndex, RefPtr<SpeechRecognitionResultList>&&);

    uint64_t m_resultIndex;
    RefPtr<SpeechRecognitionResultList> m_results;
};

} // namespace WebCore
