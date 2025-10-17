/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 28, 2023.
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
#include "WebCodecsAudioDataAlgorithms.h"

#if ENABLE(WEB_CODECS)

#include "ExceptionCode.h"
#include "ExceptionOr.h"

namespace WebCore {

// https://www.w3.org/TR/webcodecs/#valid-audiodatainit
bool isValidAudioDataInit(const WebCodecsAudioData::Init& init)
{
    if (init.sampleRate <= 0)
        return false;

    if (!init.numberOfFrames)
        return false;

    if (!init.numberOfChannels)
        return false;

    size_t totalSamples, totalSize;
    if (!WTF::safeMultiply(init.numberOfFrames, init.numberOfChannels, totalSamples))
        return false;

    auto bytesPerSample = computeBytesPerSample(init.format);
    if (!WTF::safeMultiply(totalSamples, bytesPerSample, totalSize))
        return false;

    return init.data.length() >= totalSize;
}

bool isAudioSampleFormatInterleaved(AudioSampleFormat format)
{
    switch (format) {
    case AudioSampleFormat::U8:
    case AudioSampleFormat::S16:
    case AudioSampleFormat::S32:
    case AudioSampleFormat::F32:
        return true;
    case AudioSampleFormat::U8Planar:
    case AudioSampleFormat::S16Planar:
    case AudioSampleFormat::S32Planar:
    case AudioSampleFormat::F32Planar:
        return false;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return false;
}

size_t computeBytesPerSample(AudioSampleFormat format)
{
    switch (format) {
    case AudioSampleFormat::U8:
    case AudioSampleFormat::U8Planar:
        return 1;
    case AudioSampleFormat::S16:
    case AudioSampleFormat::S16Planar:
        return 2;
    case AudioSampleFormat::S32:
    case AudioSampleFormat::F32:
    case AudioSampleFormat::S32Planar:
    case AudioSampleFormat::F32Planar:
        return 4;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return 0;
}

// https://www.w3.org/TR/webcodecs/#compute-copy-element-count
ExceptionOr<size_t> computeCopyElementCount(const WebCodecsAudioData& data, const WebCodecsAudioData::CopyToOptions& options)
{
    auto platformData = data.data().audioData;
    if (!platformData)
        return Exception { ExceptionCode::InvalidAccessError, "Internal AudioData storage is null"_s };

    // 1. Let destFormat be the value of [[format]].
    // 2. If options.format exists, assign options.format to destFormat.
    auto destFormat = options.format.value_or(platformData->format());

    // 3. If destFormat describes an interleaved AudioSampleFormat and options.planeIndex is greater than 0, throw a RangeError.
    bool isInterleaved = isAudioSampleFormatInterleaved(destFormat);
    if (isInterleaved && options.planeIndex > 0)
        return Exception { ExceptionCode::RangeError, "Invalid planeIndex for interleaved format"_s };
    // 4. Otherwise, if destFormat describes a planar AudioSampleFormat and if options.planeIndex is greater or equal to [[number of channels]], throw a RangeError.
    else if (options.planeIndex >= data.numberOfChannels())
        return Exception { ExceptionCode::RangeError, "Invalid planeIndex for planar format"_s };

#if !USE(AVFOUNDATION)
    // 5. If [[format]] does not equal destFormat and the User Agent does not support the requested AudioSampleFormat conversion, throw a NotSupportedError DOMException. Conversion to f32-planar must always be supported.
    if (options.format && *options.format != destFormat && destFormat != AudioSampleFormat::F32Planar)
        return Exception { ExceptionCode::NotSupportedError, "AudioData currently only supports copy conversion to f32-planar"_s };
#endif

    // 6. Let frameCount be the number of frames in the plane identified by options.planeIndex.
    // All planes have the same number of frames, always
    auto frameCount = data.numberOfFrames();
    // 7. If options.frameOffset is greater than or equal to frameCount, throw a RangeError.
    if (options.frameOffset && *options.frameOffset > frameCount)
        return Exception { ExceptionCode::RangeError, "frameOffset is too large"_s };

    // 8. Let copyFrameCount be the difference of subtracting options.frameOffset from frameCount.
    auto copyFrameCount = frameCount;
    if (options.frameOffset)
        copyFrameCount -= *options.frameOffset;
    // 9. If options.frameCount exists:
    if (options.frameCount) {
        // 9.1 If options.frameCount is greater than copyFrameCount, throw a RangeError.
        if (*options.frameCount > copyFrameCount)
            return Exception { ExceptionCode::RangeError, "frameCount is too large"_s };
        // 9.2 Otherwise, assign options.frameCount to copyFrameCount.
        copyFrameCount = *options.frameCount;
    }

    // 10. Let elementCount be copyFrameCount.
    // 11. If destFormat describes an interleaved AudioSampleFormat, multiply elementCount by [[number of channels]]
    if (isInterleaved) {
        size_t elementCount;
        if (!WTF::safeMultiply(copyFrameCount, data.numberOfChannels(), elementCount))
            return Exception { ExceptionCode::RangeError, "Provided options are causing an overflow"_s };

        return elementCount;
    }

    return copyFrameCount;
}

AudioSampleFormat audioSampleElementFormat(AudioSampleFormat format)
{
    switch (format) {
    case AudioSampleFormat::U8:
    case AudioSampleFormat::U8Planar:
        return AudioSampleFormat::U8;
    case AudioSampleFormat::S16:
    case AudioSampleFormat::S16Planar:
        return AudioSampleFormat::S16;
    case AudioSampleFormat::S32:
    case AudioSampleFormat::S32Planar:
        return AudioSampleFormat::S32;
    case AudioSampleFormat::F32:
    case AudioSampleFormat::F32Planar:
        return AudioSampleFormat::F32;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return AudioSampleFormat::U8;
}

AudioSampleFormatSpan audioElementSpan(AudioSampleFormat format, std::span<uint8_t> buffer)
{
    auto bytesPerSample = computeBytesPerSample(format);
    auto clampedBufferSize = buffer.size_bytes();
    clampedBufferSize -= clampedBufferSize % bytesPerSample;

    switch (format) {
    case AudioSampleFormat::U8:
    case AudioSampleFormat::U8Planar:
        return buffer;
    case AudioSampleFormat::S16:
    case AudioSampleFormat::S16Planar:
        return spanReinterpretCast<int16_t>(buffer.first(clampedBufferSize));
    case AudioSampleFormat::S32:
    case AudioSampleFormat::S32Planar:
        return spanReinterpretCast<int32_t>(buffer.first(clampedBufferSize));
    case AudioSampleFormat::F32:
    case AudioSampleFormat::F32Planar:
        return spanReinterpretCast<float>(buffer.first(clampedBufferSize));
    }
    RELEASE_ASSERT_NOT_REACHED();
    return buffer;
}

} // namespace WebCore

#endif // ENABLE(WEB_CODECS)
