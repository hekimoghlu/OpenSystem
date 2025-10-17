/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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

#if ENABLE(WEB_AUDIO)

#include "AudioBuffer.h"

#include "AudioContext.h"
#include "AudioFileReader.h"
#include "AudioUtilities.h"
#include "ScriptWrappableInlines.h"
#include "WebCoreOpaqueRoot.h"
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/TypedArrayInlines.h>
#include <wtf/CheckedArithmetic.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AudioBuffer);

RefPtr<AudioBuffer> AudioBuffer::create(unsigned numberOfChannels, size_t numberOfFrames, float sampleRate, LegacyPreventDetaching preventDetaching)
{
    if (!BaseAudioContext::isSupportedSampleRate(sampleRate) || !numberOfChannels || numberOfChannels > AudioContext::maxNumberOfChannels || !numberOfFrames)
        return nullptr;

    auto buffer = adoptRef(*new AudioBuffer(numberOfChannels, numberOfFrames, sampleRate, preventDetaching));
    if (!buffer->originalLength())
        return nullptr;

    return buffer;
}

ExceptionOr<Ref<AudioBuffer>> AudioBuffer::create(const AudioBufferOptions& options)
{
    if (!options.numberOfChannels)
        return Exception { ExceptionCode::NotSupportedError, "Number of channels cannot be 0."_s };

    if (options.numberOfChannels > AudioContext::maxNumberOfChannels)
        return Exception { ExceptionCode::NotSupportedError, "Number of channels cannot be more than max supported."_s };
    
    if (!options.length)
        return Exception { ExceptionCode::NotSupportedError, "Length must be at least 1."_s };
    
    if (!BaseAudioContext::isSupportedSampleRate(options.sampleRate))
        return Exception { ExceptionCode::NotSupportedError, "Sample rate is not in the supported range."_s };
    
    auto buffer = adoptRef(*new AudioBuffer(options.numberOfChannels, options.length, options.sampleRate));
    if (!buffer->originalLength())
        return Exception { ExceptionCode::NotSupportedError, "Channel was not able to be created."_s };
    
    return buffer;
}

RefPtr<AudioBuffer> AudioBuffer::createFromAudioFileData(std::span<const uint8_t> data, bool mixToMono, float sampleRate)
{
    RefPtr bus = createBusFromInMemoryAudioFile(data, mixToMono, sampleRate);
    if (!bus)
        return nullptr;
    return adoptRef(*new AudioBuffer(*bus));
}

AudioBuffer::AudioBuffer(unsigned numberOfChannels, size_t length, float sampleRate, LegacyPreventDetaching preventDetaching)
    : m_sampleRate(sampleRate)
    , m_originalLength(length)
    , m_isDetachable(preventDetaching == LegacyPreventDetaching::No)
{
    auto totalLength = CheckedUint64(m_originalLength) * numberOfChannels;
    if (totalLength.hasOverflowed() || totalLength.value() > s_maxLength || static_cast<uint64_t>(m_originalLength) > s_maxChannelLength) {
        invalidate();
        return;
    }

    Vector<RefPtr<Float32Array>> channels;
    channels.reserveInitialCapacity(numberOfChannels);

    for (unsigned i = 0; i < numberOfChannels; ++i) {
        auto channelDataArray = Float32Array::tryCreate(m_originalLength);
        if (!channelDataArray) {
            invalidate();
            return;
        }

        if (preventDetaching == LegacyPreventDetaching::Yes)
            channelDataArray->setDetachable(false);

        channels.append(WTFMove(channelDataArray));
    }

    m_channels = WTFMove(channels);
    m_channelWrappers = FixedVector<JSValueInWrappedObject> { m_channels.size() };
}

AudioBuffer::AudioBuffer(AudioBus& bus)
    : m_sampleRate(bus.sampleRate())
    , m_originalLength(bus.length())
{
    unsigned numberOfChannels = bus.numberOfChannels();
    auto totalLength = CheckedUint64(m_originalLength) * numberOfChannels;
    if (totalLength.hasOverflowed() || totalLength.value() > s_maxLength || static_cast<uint64_t>(m_originalLength) > s_maxChannelLength) {
        invalidate();
        return;
    }

    // Copy audio data from the bus to the Float32Arrays we manage.
    Vector<RefPtr<Float32Array>> channels;
    channels.reserveInitialCapacity(numberOfChannels);
    for (unsigned i = 0; i < numberOfChannels; ++i) {
        auto channelDataArray = Float32Array::tryCreate(m_originalLength);
        if (!channelDataArray) {
            invalidate();
            return;
        }

        channelDataArray->setRange(bus.channel(i)->data(), m_originalLength, 0);
        channels.append(WTFMove(channelDataArray));
    }

    m_channels = WTFMove(channels);
    m_channelWrappers = FixedVector<JSValueInWrappedObject> { m_channels.size() };
}

// FIXME: We don't currently implement "acquire the content" [1] and
// AudioBuffer::getChannelData() correctly. As a result, the audio thread
// may be reading an array buffer that the JS can detach. To guard against
// this, we mark the array buffers as non-detachable as soon as their content
// has been acquired.
// [1] https://www.w3.org/TR/webaudio/#acquire-the-content
void AudioBuffer::markBuffersAsNonDetachable()
{
    Locker locker { m_channelsLock };
    for (auto& channel : m_channels)
        channel->setDetachable(false);
}

void AudioBuffer::invalidate()
{
    releaseMemory();
    m_originalLength = 0;
}

void AudioBuffer::releaseMemory()
{
    Locker locker { m_channelsLock };
    m_channels = { };
    m_channelWrappers = { };
    m_noiseInjectionMultiplier = 0;
}

ExceptionOr<JSC::JSValue> AudioBuffer::getChannelData(JSDOMGlobalObject& globalObject, unsigned channelIndex)
{
    ASSERT(m_channelWrappers.size() == m_channels.size());
    if (channelIndex >= m_channelWrappers.size())
        return Exception { ExceptionCode::IndexSizeError, "Index must be less than number of channels."_s };

    applyNoiseIfNeeded();

    auto& channelData = m_channels[channelIndex];
    auto constructJSArray = [&] {
        constexpr bool isResizableOrGrowableShared = false;
        return JSC::JSFloat32Array::create(globalObject.vm(), globalObject.typedArrayStructure(JSC::TypeFloat32, isResizableOrGrowableShared), channelData.copyRef());
    };

    if (globalObject.worldIsNormal()) {
        if (!m_channelWrappers[channelIndex])
            m_channelWrappers[channelIndex].set(globalObject.vm(), wrapper(), constructJSArray());
        return m_channelWrappers[channelIndex].getValue();
    }
    return constructJSArray();
}

template<typename Visitor>
void AudioBuffer::visitChannelWrappers(Visitor& visitor)
{
    Locker locker { m_channelsLock };
    for (auto& channelWrapper : m_channelWrappers)
        channelWrapper.visit(visitor);
}

template void AudioBuffer::visitChannelWrappers(JSC::AbstractSlotVisitor&);
template void AudioBuffer::visitChannelWrappers(JSC::SlotVisitor&);

RefPtr<Float32Array> AudioBuffer::channelData(unsigned channelIndex)
{
    if (channelIndex >= m_channels.size())
        return nullptr;
    if (hasDetachedChannelBuffer())
        return Float32Array::create(0);
    return m_channels[channelIndex].copyRef();
}

std::span<float> AudioBuffer::rawChannelData(unsigned channelIndex)
{
    if (channelIndex >= m_channels.size())
        return { };
    if (hasDetachedChannelBuffer())
        return { };
    return m_channels[channelIndex]->typedMutableSpan();
}

ExceptionOr<void> AudioBuffer::copyFromChannel(Ref<Float32Array>&& destination, unsigned channelNumber, unsigned bufferOffset)
{
    if (destination->isShared())
        return Exception { ExceptionCode::TypeError, "Destination may not be a shared buffer."_s };
    
    if (channelNumber >= m_channels.size())
        return Exception { ExceptionCode::IndexSizeError, "Not a valid channelNumber."_s };
    
    Float32Array* channelData = m_channels[channelNumber].get();
    
    size_t dataLength = channelData->length();
    
    if (bufferOffset >= dataLength)
        return { };
    
    applyNoiseIfNeeded();

    size_t count = dataLength - bufferOffset;
    count = std::min(destination.get().length(), count);
    
    auto src = channelData->typedSpan();
    auto dst = destination->typedMutableSpan();
    
    ASSERT(src.data());
    ASSERT(dst.data());
    
    memmoveSpan(dst, src.subspan(bufferOffset, count));
    return { };
}

ExceptionOr<void> AudioBuffer::copyToChannel(Ref<Float32Array>&& source, unsigned channelNumber, unsigned bufferOffset)
{
    if (source->isShared())
        return Exception { ExceptionCode::TypeError, "Source may not be a shared buffer."_s };
    
    if (channelNumber >= m_channels.size())
        return Exception { ExceptionCode::IndexSizeError, "Not a valid channelNumber."_s };
    
    Float32Array* channelData = m_channels[channelNumber].get();
    
    size_t dataLength = channelData->length();
    
    if (bufferOffset >= dataLength)
        return { };
    
    size_t count = dataLength - bufferOffset;
    count = std::min(source.get().length(), count);
    
    auto src = source->typedSpan();
    auto dst = channelData->typedMutableSpan();
    
    ASSERT(src.data());
    ASSERT(dst.data());
    
    memmoveSpan(dst.subspan(bufferOffset), src.first(count));
    m_noiseInjectionMultiplier = 0;
    return { };
}

void AudioBuffer::zero()
{
    for (auto& channel : m_channels)
        channel->zeroFill();

    m_noiseInjectionMultiplier = 0;
}

size_t AudioBuffer::memoryCost() const
{
    // memoryCost() may be invoked concurrently from a GC thread, and we need to be careful
    // about what data we access here and how. We need to hold a lock to prevent m_channels
    // from being changed while we iterate it, but calling channel->byteLength() is safe
    // because it doesn't involve chasing any pointers that can be nullified while the
    // AudioBuffer is alive.
    Locker locker { m_channelsLock };
    size_t cost = 0;
    for (auto& channel : m_channels)
        cost += channel->byteLength();
    return cost;
}

bool AudioBuffer::hasDetachedChannelBuffer() const
{
    for (auto& channel : m_channels) {
        if (channel->isDetached())
            return true;
    }
    return false;
}

bool AudioBuffer::topologyMatches(const AudioBuffer& other) const
{
    return numberOfChannels() == other.numberOfChannels() && length() == other.length() && sampleRate() == other.sampleRate();
}

bool AudioBuffer::copyTo(AudioBuffer& other) const
{
    if (!topologyMatches(other))
        return false;

    if (hasDetachedChannelBuffer() || other.hasDetachedChannelBuffer())
        return false;

    for (unsigned channelIndex = 0; channelIndex < numberOfChannels(); ++channelIndex)
        memcpySpan(other.rawChannelData(channelIndex), m_channels[channelIndex]->typedSpan().first(length()));

    other.m_noiseInjectionMultiplier = m_noiseInjectionMultiplier;
    return true;
}

Ref<AudioBuffer> AudioBuffer::clone(ShouldCopyChannelData shouldCopyChannelData) const
{
    auto clone = AudioBuffer::create(numberOfChannels(), length(), sampleRate(), m_isDetachable ? LegacyPreventDetaching::No : LegacyPreventDetaching::Yes);
    ASSERT(clone);
    if (shouldCopyChannelData == ShouldCopyChannelData::Yes)
        copyTo(*clone);
    return clone.releaseNonNull();
}

WebCoreOpaqueRoot root(AudioBuffer* buffer)
{
    return WebCoreOpaqueRoot { buffer };
}

void AudioBuffer::applyNoiseIfNeeded()
{
    if (!m_noiseInjectionMultiplier)
        return;

    for (auto& channel : m_channels)
        AudioUtilities::applyNoise(channel->typedMutableSpan(), m_noiseInjectionMultiplier);

    m_noiseInjectionMultiplier = 0;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
