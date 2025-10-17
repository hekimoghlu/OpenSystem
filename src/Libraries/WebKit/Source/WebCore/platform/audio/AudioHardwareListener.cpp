/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#include "AudioHardwareListener.h"

#include <wtf/Function.h>
#include <wtf/NeverDestroyed.h>

#if PLATFORM(MAC)
#include "AudioHardwareListenerMac.h"
#endif

namespace WebCore {

static AudioHardwareListener::CreationFunction& audioHardwareListenerCreationFunction()
{
    static NeverDestroyed<AudioHardwareListener::CreationFunction> creationFunction;
    return creationFunction;
}

void AudioHardwareListener::setCreationFunction(CreationFunction&& function)
{
    audioHardwareListenerCreationFunction() = WTFMove(function);
}

void AudioHardwareListener::resetCreationFunction()
{
    audioHardwareListenerCreationFunction() = [] (AudioHardwareListener::Client& client) {
#if PLATFORM(MAC)
        return AudioHardwareListenerMac::create(client);
#else
        class RefCountedAudioHardwareListener : public AudioHardwareListener, public RefCounted<RefCountedAudioHardwareListener> {
        public:
            void ref() const final { RefCounted::ref(); }
            void deref() const final { RefCounted::deref(); }

            RefCountedAudioHardwareListener(AudioHardwareListener::Client& client)
                : AudioHardwareListener(client) { }
        };
        return adoptRef(*new RefCountedAudioHardwareListener(client));
#endif
    };
}

Ref<AudioHardwareListener> AudioHardwareListener::create(Client& client)
{
    if (!audioHardwareListenerCreationFunction())
        resetCreationFunction();

    return audioHardwareListenerCreationFunction()(client);
}

AudioHardwareListener::AudioHardwareListener(Client& client)
    : m_client(client)
{
#if PLATFORM(IOS_FAMILY)
    m_supportedBufferSizes = { 32, 4096 };
#endif
}

}
