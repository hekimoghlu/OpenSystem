/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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

#if ENABLE(WEB_AUDIO)

#include <CoreAudio/CoreAudioTypes.h>
#include <optional>
#include <wtf/LoggerHelper.h>
#include <wtf/RefPtr.h>

using ExtAudioFileRef = struct OpaqueExtAudioFile*;
using AudioFileID = struct OpaqueAudioFileID*;
typedef struct opaqueCMSampleBuffer* CMSampleBufferRef;

namespace WebCore {

class AudioBus;
class SourceBufferParserWebM;
class AudioFileReaderWebMData;

// Wrapper class for AudioFile and ExtAudioFile CoreAudio APIs for reading files and in-memory versions of them...

class AudioFileReader
#if !RELEASE_LOG_DISABLED
    : public LoggerHelper
#endif
{
public:
    explicit AudioFileReader(std::span<const uint8_t> data);
    ~AudioFileReader();

    RefPtr<AudioBus> createBus(float sampleRate, bool mixToMono); // Returns nullptr on error

    size_t dataSize() const { return m_data.size(); }
    std::span<const uint8_t> span() const { return m_data; }

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    WTFLogChannel& logChannel() const final;
    ASCIILiteral logClassName() const final { return "AudioFileReaderCocoa"_s; }
#endif

private:
#if ENABLE(MEDIA_SOURCE)
    bool isMaybeWebM(std::span<const uint8_t>) const;
    std::unique_ptr<AudioFileReaderWebMData> demuxWebMData(std::span<const uint8_t>) const;
    std::optional<size_t> decodeWebMData(AudioBufferList&, size_t numberOfFrames, const AudioStreamBasicDescription& inFormat, const AudioStreamBasicDescription& outFormat) const;
#endif
    static OSStatus readProc(void* clientData, SInt64 position, UInt32 requestCount, void* buffer, UInt32* actualCount);
    static SInt64 getSizeProc(void* clientData);
    ssize_t numberOfFrames() const;
    std::optional<AudioStreamBasicDescription> fileDataFormat() const;
    AudioStreamBasicDescription clientDataFormat(const AudioStreamBasicDescription& inFormat, float sampleRate) const;

    std::span<const uint8_t> m_data;

    AudioFileID m_audioFileID = { nullptr };
    ExtAudioFileRef m_extAudioFileRef = { nullptr };

    std::unique_ptr<AudioFileReaderWebMData> m_webmData;

#if !RELEASE_LOG_DISABLED
    const Ref<Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif

};

}

#endif // ENABLE(WEB_AUDIO)
