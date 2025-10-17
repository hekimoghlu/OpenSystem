/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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
#ifndef AudioResampler_h
#define AudioResampler_h

#include "AudioBus.h"
#include "AudioResamplerKernel.h"
#include "AudioSourceProvider.h"
#include <memory>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

// AudioResampler resamples the audio stream from an AudioSourceProvider.
// The audio stream may be single or multi-channel.
// The default constructor defaults to single-channel (mono).

class AudioResampler final {
    WTF_MAKE_TZONE_ALLOCATED(AudioResampler);
public:
    AudioResampler();
    AudioResampler(unsigned numberOfChannels);
    ~AudioResampler() = default;
    
    // Given an AudioSourceProvider, process() resamples the source stream into destinationBus.
    void process(AudioSourceProvider*, AudioBus* destinationBus, size_t framesToProcess);

    // Resets the processing state.
    void reset();

    void configureChannels(unsigned numberOfChannels);

    // 0 < rate <= MaxRate
    void setRate(double rate);
    double rate() const { return m_rate; }

    static constexpr double MaxRate { 8 };

private:
    double m_rate { 1 };
    Vector<std::unique_ptr<AudioResamplerKernel>> m_kernels;
    RefPtr<AudioBus> m_sourceBus;
};

} // namespace WebCore

#endif // AudioResampler_h
