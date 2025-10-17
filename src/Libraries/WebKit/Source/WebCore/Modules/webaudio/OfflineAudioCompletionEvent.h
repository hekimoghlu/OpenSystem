/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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

class AudioBuffer;
struct OfflineAudioCompletionEventInit;

class OfflineAudioCompletionEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(OfflineAudioCompletionEvent);
public:
    static Ref<OfflineAudioCompletionEvent> create(Ref<AudioBuffer>&& renderedBuffer);
    static Ref<OfflineAudioCompletionEvent> create(const AtomString& eventType, OfflineAudioCompletionEventInit&&);
    
    virtual ~OfflineAudioCompletionEvent();

    AudioBuffer& renderedBuffer() { return m_renderedBuffer.get(); }

private:
    explicit OfflineAudioCompletionEvent(Ref<AudioBuffer>&& renderedBuffer);
    OfflineAudioCompletionEvent(const AtomString& eventType, OfflineAudioCompletionEventInit&&);

    Ref<AudioBuffer> m_renderedBuffer;
};

} // namespace WebCore
