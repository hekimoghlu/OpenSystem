/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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

#include "AudioBuffer.h"
#include "Event.h"
#include <wtf/RefPtr.h>

namespace WebCore {

class AudioBuffer;
struct AudioProcessingEventInit;
    
class AudioProcessingEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AudioProcessingEvent);
public:
    static Ref<AudioProcessingEvent> create(RefPtr<AudioBuffer>&& inputBuffer, RefPtr<AudioBuffer>&& outputBuffer, double playbackTime)
    {
        return adoptRef(*new AudioProcessingEvent(WTFMove(inputBuffer), WTFMove(outputBuffer), playbackTime));
    }
    
    static Ref<AudioProcessingEvent> create(const AtomString&, AudioProcessingEventInit&&);
    
    virtual ~AudioProcessingEvent();

    AudioBuffer* inputBuffer() { return m_inputBuffer.get(); }
    AudioBuffer* outputBuffer() { return m_outputBuffer.get(); }
    double playbackTime() const { return m_playbackTime; }

private:
    AudioProcessingEvent(RefPtr<AudioBuffer>&& inputBuffer, RefPtr<AudioBuffer>&& outputBuffer, double playbackTime);
    AudioProcessingEvent(const AtomString&, AudioProcessingEventInit&&);

    RefPtr<AudioBuffer> m_inputBuffer;
    RefPtr<AudioBuffer> m_outputBuffer;
    double m_playbackTime;
};

} // namespace WebCore
