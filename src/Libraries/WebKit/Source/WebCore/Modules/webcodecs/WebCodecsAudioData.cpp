/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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
#include "WebCodecsAudioData.h"

#if ENABLE(WEB_CODECS)

#include "ContextDestructionObserverInlines.h"
#include "ExceptionOr.h"
#include "JSDOMPromiseDeferred.h"
#include "PlatformRawAudioData.h"
#include "WebCodecsAudioDataAlgorithms.h"

namespace WebCore {

// https://www.w3.org/TR/webcodecs/#dom-audiodata-audiodata
ExceptionOr<Ref<WebCodecsAudioData>> WebCodecsAudioData::create(ScriptExecutionContext& context, Init&& init)
{
    if (!isValidAudioDataInit(init))
        return Exception { ExceptionCode::TypeError, "Invalid init data"_s };

    auto rawData = init.data.span();
    auto data = PlatformRawAudioData::create(WTFMove(rawData), init.format, init.sampleRate, init.timestamp, init.numberOfFrames, init.numberOfChannels);

    if (!data)
        return Exception { ExceptionCode::NotSupportedError, "AudioData creation failed"_s };

    return adoptRef(*new WebCodecsAudioData(context, WebCodecsAudioInternalData { WTFMove(data) }));
}

Ref<WebCodecsAudioData> WebCodecsAudioData::create(ScriptExecutionContext& context, Ref<PlatformRawAudioData>&& data)
{
    return adoptRef(*new WebCodecsAudioData(context, WebCodecsAudioInternalData { WTFMove(data) }));
}

WebCodecsAudioData::WebCodecsAudioData(ScriptExecutionContext& context)
    : ContextDestructionObserver(&context)
{
}

WebCodecsAudioData::WebCodecsAudioData(ScriptExecutionContext& context, WebCodecsAudioInternalData&& data)
    : ContextDestructionObserver(&context)
    , m_data(WTFMove(data))
{
}

WebCodecsAudioData::~WebCodecsAudioData() = default;

std::optional<AudioSampleFormat> WebCodecsAudioData::format() const
{
    return m_data.audioData ? std::make_optional(m_data.audioData->format()) : std::nullopt;
}

float WebCodecsAudioData::sampleRate() const
{
    return m_data.audioData ? m_data.audioData->sampleRate() : 0;
}

size_t WebCodecsAudioData::numberOfFrames() const
{
    return m_data.audioData ? m_data.audioData->numberOfFrames() : 0;
}

size_t WebCodecsAudioData::numberOfChannels() const
{
    return m_data.audioData ? m_data.audioData->numberOfChannels() : 0;
}

std::optional<uint64_t> WebCodecsAudioData::duration()
{
    return m_data.audioData ? m_data.audioData->duration() : std::nullopt;
}

int64_t WebCodecsAudioData::timestamp() const
{
    return m_data.audioData ? m_data.audioData->timestamp() : 0;
}

// https://www.w3.org/TR/webcodecs/#dom-audiodata-allocationsize
ExceptionOr<size_t> WebCodecsAudioData::allocationSize(const CopyToOptions& options)
{
    if (isDetached())
        return Exception { ExceptionCode::InvalidStateError, "AudioData is detached"_s };

    auto copyElementCount = computeCopyElementCount(*this, options);
    if (copyElementCount.hasException())
        return copyElementCount.releaseException();

    auto destFormat = options.format.value_or(*format());
    auto bytesPerSample = computeBytesPerSample(destFormat);
    return copyElementCount.releaseReturnValue() * bytesPerSample;
}

// https://www.w3.org/TR/webcodecs/#dom-audiodata-copyto
ExceptionOr<void> WebCodecsAudioData::copyTo(BufferSource&& source, CopyToOptions&& options)
{
    if (isDetached())
        return Exception { ExceptionCode::InvalidStateError, "AudioData is detached"_s };

    auto copyElementCount = computeCopyElementCount(*this, options);
    if (copyElementCount.hasException())
        return copyElementCount.releaseException();

    auto destFormat = options.format.value_or(*format());
    auto bytesPerSample = computeBytesPerSample(destFormat);
    auto maxCopyElementCount = copyElementCount.releaseReturnValue();
    unsigned long allocationSize;
    if (!WTF::safeMultiply(maxCopyElementCount, bytesPerSample, allocationSize))
        return Exception { ExceptionCode::RangeError, "Calculated destination buffer size overflows"_s };

    if (allocationSize > source.length())
        return Exception { ExceptionCode::RangeError, "Buffer is too small"_s };

    m_data.audioData->copyTo(source.mutableSpan(), destFormat, options.planeIndex, options.frameOffset, options.frameCount, maxCopyElementCount);
    return { };
}

// https://www.w3.org/TR/webcodecs/#dom-audiodata-clone
ExceptionOr<Ref<WebCodecsAudioData>> WebCodecsAudioData::clone(ScriptExecutionContext& context)
{
    if (isDetached())
        return Exception { ExceptionCode::InvalidStateError,  "AudioData is detached"_s };

    return adoptRef(*new WebCodecsAudioData(context, WebCodecsAudioInternalData { m_data }));
}

// https://www.w3.org/TR/webcodecs/#dom-audiodata-close
void WebCodecsAudioData::close()
{
    m_data.audioData = nullptr;

    m_isDetached = true;
}

} // namespace WebCore

#endif // ENABLE(WEB_CODECS)
