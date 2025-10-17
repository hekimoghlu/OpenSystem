/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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

#include "AudioBufferOptions.h"
#include "ExceptionOr.h"
#include "JSValueInWrappedObject.h"
#include "ScriptWrappable.h"
#include <JavaScriptCore/Forward.h>
#include <JavaScriptCore/TypedArrayAdaptersForwardDeclarations.h>
#include <wtf/Lock.h>
#include <wtf/Vector.h>

namespace WebCore {

class AudioBus;
class WebCoreOpaqueRoot;

class AudioBuffer : public ScriptWrappable, public RefCounted<AudioBuffer> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AudioBuffer);
public:
    enum class LegacyPreventDetaching : bool { No, Yes };
    static RefPtr<AudioBuffer> create(unsigned numberOfChannels, size_t numberOfFrames, float sampleRate, LegacyPreventDetaching = LegacyPreventDetaching::No);
    static ExceptionOr<Ref<AudioBuffer>> create(const AudioBufferOptions&);
    // Returns nullptr if data is not a valid audio file.
    static RefPtr<AudioBuffer> createFromAudioFileData(std::span<const uint8_t> data, bool mixToMono, float sampleRate);

    // Format
    size_t originalLength() const { return m_originalLength; }
    double originalDuration() const { return originalLength() / static_cast<double>(sampleRate()); }
    float sampleRate() const { return m_sampleRate; }

    // The following function may start returning 0 if any of the underlying channel buffers gets detached.
    size_t length() const { return hasDetachedChannelBuffer() ? 0 : m_originalLength; }
    double duration() const { return length() / static_cast<double>(sampleRate()); }

    void markBuffersAsNonDetachable();

    // Channel data access
    unsigned numberOfChannels() const { return m_channels.size(); }
    ExceptionOr<JSC::JSValue> getChannelData(JSDOMGlobalObject&, unsigned channelIndex);
    ExceptionOr<void> copyFromChannel(Ref<Float32Array>&&, unsigned channelNumber, unsigned bufferOffset);
    ExceptionOr<void> copyToChannel(Ref<Float32Array>&&, unsigned channelNumber, unsigned startInChannel);

    // Native channel data access.
    RefPtr<Float32Array> channelData(unsigned channelIndex);
    std::span<float> rawChannelData(unsigned channelIndex);
    void zero();

    // Because an AudioBuffer has a JavaScript wrapper, which will be garbage collected, it may take a while for this object to be deleted.
    // releaseMemory() can be called when the AudioContext goes away, so we can release the memory earlier than when the garbage collection happens.
    // Careful! Only call this when the page unloads, after the AudioContext is no longer processing.
    void releaseMemory();

    size_t memoryCost() const;

    template<typename Visitor> void visitChannelWrappers(Visitor&);

    bool copyTo(AudioBuffer&) const;

    enum class ShouldCopyChannelData : bool { No, Yes };
    Ref<AudioBuffer> clone(ShouldCopyChannelData = ShouldCopyChannelData::Yes) const;
    
    bool topologyMatches(const AudioBuffer&) const;

    void increaseNoiseInjectionMultiplier(float amount = 0.001) { m_noiseInjectionMultiplier += amount; }
    float noiseInjectionMultiplier() const { return m_noiseInjectionMultiplier; }

private:
    AudioBuffer(unsigned numberOfChannels, size_t length, float sampleRate, LegacyPreventDetaching = LegacyPreventDetaching::No);
    explicit AudioBuffer(AudioBus&);

    void invalidate();

    bool hasDetachedChannelBuffer() const;

    void applyNoiseIfNeeded();

    // We do not currently support having the Float32Arrays in m_channels being more than 2GB,
    // and we have tests that we return an error promptly on trying to create such a huge AudioBuffer.
    static constexpr uint64_t s_maxChannelLength = (1ull << 32) / sizeof(float);
    static constexpr uint64_t s_maxLength = 1ull << 32;

    float m_sampleRate;
    size_t m_originalLength;
    FixedVector<RefPtr<Float32Array>> m_channels;
    FixedVector<JSValueInWrappedObject> m_channelWrappers;
    bool m_isDetachable { true };
    mutable Lock m_channelsLock;
    float m_noiseInjectionMultiplier { 0 };
};

WebCoreOpaqueRoot root(AudioBuffer*);

} // namespace WebCore
