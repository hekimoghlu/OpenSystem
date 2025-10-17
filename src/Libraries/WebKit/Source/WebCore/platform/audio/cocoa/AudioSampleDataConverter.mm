/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#import "AudioSampleDataConverter.h"

#import "AudioSampleBufferList.h"
#import "DeprecatedGlobalSettings.h"
#import "Logging.h"
#import <AudioToolbox/AudioConverter.h>
#import <wtf/TZoneMallocInlines.h>

#import <pal/cf/AudioToolboxSoftLink.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioSampleDataConverter);

AudioSampleDataConverter::AudioSampleDataConverter()
    : m_latencyAdaptationEnabled(DeprecatedGlobalSettings::webRTCAudioLatencyAdaptationEnabled())
{
}

AudioSampleDataConverter::~AudioSampleDataConverter()
{
}

OSStatus AudioSampleDataConverter::setFormats(const CAAudioStreamDescription& inputDescription, const CAAudioStreamDescription& outputDescription)
{
    constexpr double buffer100ms = 0.100;
    constexpr double buffer60ms = 0.060;
    constexpr double buffer50ms = 0.050;
    constexpr double buffer40ms = 0.040;
    constexpr double buffer20ms = 0.020;
    m_highBufferSize = outputDescription.sampleRate() * buffer100ms;
    m_regularHighBufferSize = outputDescription.sampleRate() * buffer60ms;
    m_regularBufferSize = outputDescription.sampleRate() * buffer50ms;
    m_regularLowBufferSize = outputDescription.sampleRate() * buffer40ms;
    m_lowBufferSize = outputDescription.sampleRate() * buffer20ms;

    m_selectedConverter = nullptr;

    auto converterOutputDescription = outputDescription.streamDescription();
    constexpr double slightlyHigherPitch = 1.05;
    converterOutputDescription.mSampleRate = slightlyHigherPitch * outputDescription.streamDescription().mSampleRate;
    if (auto error = m_lowConverter.initialize(inputDescription.streamDescription(), converterOutputDescription); error != noErr)
        return error;

    constexpr double slightlyLowerPitch = 0.95;
    converterOutputDescription.mSampleRate = slightlyLowerPitch * outputDescription.streamDescription().mSampleRate;
    if (auto error = m_highConverter.initialize(inputDescription.streamDescription(), converterOutputDescription); error != noErr)
        return error;

    if (inputDescription == outputDescription)
        return noErr;

    if (auto error = m_regularConverter.initialize(inputDescription.streamDescription(), outputDescription.streamDescription()); error != noErr)
        return error;

    m_selectedConverter = m_regularConverter;
    return noErr;
}

bool AudioSampleDataConverter::updateBufferedAmount(size_t currentBufferedAmount, size_t pushedSampleSize)
{
    if (!m_latencyAdaptationEnabled)
        return !!m_selectedConverter;

    if (currentBufferedAmount) {
        DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;
        if (m_selectedConverter == m_regularConverter) {
            if (currentBufferedAmount <= m_lowBufferSize) {
                m_selectedConverter = m_lowConverter;
                callOnMainThread([] {
                    RELEASE_LOG(WebRTC, "AudioSampleDataConverter::updateBufferedAmount low buffer");
                });
            } else if (currentBufferedAmount >= m_highBufferSize && currentBufferedAmount >= 4 * pushedSampleSize) {
                m_selectedConverter = m_highConverter;
                callOnMainThread([] {
                    RELEASE_LOG(WebRTC, "AudioSampleDataConverter::updateBufferedAmount high buffer");
                });
            }
        } else if (m_selectedConverter == m_highConverter) {
            if (currentBufferedAmount < m_regularLowBufferSize) {
                m_selectedConverter = m_regularConverter;
                callOnMainThread([] {
                    RELEASE_LOG(WebRTC, "AudioSampleDataConverter::updateBufferedAmount going down to regular buffer");
                });
            }
        } else if (currentBufferedAmount > m_regularHighBufferSize) {
            m_selectedConverter = m_regularConverter;
            callOnMainThread([] {
                RELEASE_LOG(WebRTC, "AudioSampleDataConverter::updateBufferedAmount going up to regular buffer");
            });
        }
    }
    return !!m_selectedConverter;
}

OSStatus AudioSampleDataConverter::convert(const AudioBufferList& inputBuffer, AudioSampleBufferList& outputBuffer, size_t sampleCount)
{
    outputBuffer.reset();
    return outputBuffer.copyFrom(inputBuffer, sampleCount, m_selectedConverter);
}

OSStatus AudioSampleDataConverter::Converter::initialize(const AudioStreamBasicDescription& inputDescription, const AudioStreamBasicDescription& outputDescription)
{
    DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;
    if (m_audioConverter) {
        PAL::AudioConverterDispose(m_audioConverter);
        m_audioConverter = nullptr;
    }

    return PAL::AudioConverterNew(&inputDescription, &outputDescription, &m_audioConverter);
}

AudioSampleDataConverter::Converter::~Converter()
{
    if (m_audioConverter)
        PAL::AudioConverterDispose(m_audioConverter);
}

} // namespace WebCore
