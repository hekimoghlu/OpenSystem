/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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

#include <wtf/TZoneMalloc.h>

typedef struct AudioBufferList AudioBufferList;
struct AudioStreamBasicDescription;
typedef struct OpaqueAudioConverter* AudioConverterRef;

namespace WebCore {

class AudioSampleBufferList;
class CAAudioStreamDescription;
class PlatformAudioData;

class AudioSampleDataConverter {
    WTF_MAKE_TZONE_ALLOCATED(AudioSampleDataConverter);
public:
    AudioSampleDataConverter();
    ~AudioSampleDataConverter();

    OSStatus setFormats(const CAAudioStreamDescription& inputDescription, const CAAudioStreamDescription& outputDescription);
    bool updateBufferedAmount(size_t currentBufferedAmount, size_t pushedSampleSize);
    OSStatus convert(const AudioBufferList&, AudioSampleBufferList&, size_t sampleCount);
    size_t regularBufferSize() const { return m_regularBufferSize; }
    bool isRegular() const { return m_selectedConverter == m_regularConverter; }

private:
    size_t m_highBufferSize { 0 };
    size_t m_regularHighBufferSize { 0 };
    size_t m_regularBufferSize { 0 };
    size_t m_regularLowBufferSize { 0 };
    size_t m_lowBufferSize { 0 };

    class Converter {
    public:
        Converter() = default;
        ~Converter();

        OSStatus initialize(const AudioStreamBasicDescription& inputDescription, const AudioStreamBasicDescription& outputDescription);
        operator AudioConverterRef() const { return m_audioConverter; }

    private:
        AudioConverterRef m_audioConverter { nullptr };
    };

    bool m_latencyAdaptationEnabled { true };
    Converter m_lowConverter;
    Converter m_regularConverter;
    Converter m_highConverter;
    AudioConverterRef m_selectedConverter;
};

} // namespace WebCore
