/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#import "config.h"
#import "SpeechRecognizer.h"

#if HAVE(SPEECHRECOGNIZER)

#import "AudioStreamDescription.h"
#import "MediaUtilities.h"
#import "SpeechRecognitionUpdate.h"
#import "WebSpeechRecognizerTaskMock.h"
#import <Speech/Speech.h>
#import <pal/avfoundation/MediaTimeAVFoundation.h>
#import <pal/cf/CoreMediaSoftLink.h>

namespace WebCore {

void SpeechRecognizer::dataCaptured(const MediaTime&, const PlatformAudioData& data, const AudioStreamDescription& description, size_t sampleCount)
{
    auto buffer = createAudioSampleBuffer(data, description, m_currentAudioSampleTime, sampleCount);
    [m_task audioSamplesAvailable:buffer.get()];
    m_currentAudioSampleTime = PAL::CMTimeAdd(m_currentAudioSampleTime, PAL::toCMTime(MediaTime(sampleCount, description.sampleRate())));
}

bool SpeechRecognizer::startRecognition(bool mockSpeechRecognitionEnabled, SpeechRecognitionConnectionClientIdentifier identifier, const String& localeIdentifier, bool continuous, bool interimResults, uint64_t alternatives)
{
    auto taskClass = mockSpeechRecognitionEnabled ? [WebSpeechRecognizerTaskMock class] : [WebSpeechRecognizerTask class];
    m_task = adoptNS([[taskClass alloc] initWithIdentifier:identifier locale:localeIdentifier doMultipleRecognitions:continuous reportInterimResults:interimResults maxAlternatives:alternatives delegateCallback:[weakThis = WeakPtr { *this }](const WebCore::SpeechRecognitionUpdate& update) {
        if (weakThis)
            weakThis->m_delegateCallback(update);
    }]);

    return !!m_task;
}

void SpeechRecognizer::stopRecognition()
{
    ASSERT(m_task);
    [m_task stop];
}

void SpeechRecognizer::abortRecognition()
{
    ASSERT(m_task);
    [m_task abort];
}

} // namespace WebCore

#endif // HAVE(SPEECHRECOGNIZER)
