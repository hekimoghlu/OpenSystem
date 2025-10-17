/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
#ifndef MultiChannelResampler_h
#define MultiChannelResampler_h

#include <memory>
#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class AudioBus;
class SincResampler;

class MultiChannelResampler final {
    WTF_MAKE_TZONE_ALLOCATED(MultiChannelResampler);
public:   
    // requestFrames constrols the size of the buffer in frames when provideInput is called.
    MultiChannelResampler(double scaleFactor, unsigned numberOfChannels, unsigned requestFrames, Function<void(AudioBus*, size_t framesToProcess)>&& provideInput);
    ~MultiChannelResampler();

    void process(AudioBus* destination, size_t framesToProcess);

private:
    void provideInputForChannel(std::span<float> buffer, size_t framesToProcess, unsigned channelIndex);

    // FIXME: the mac port can have a more highly optimized implementation based on CoreAudio
    // instead of SincResampler. For now the default implementation will be used on all ports.
    // https://bugs.webkit.org/show_bug.cgi?id=75118
    
    // Each channel will be resampled using a high-quality SincResampler.
    Vector<std::unique_ptr<SincResampler>> m_kernels;
    
    unsigned m_numberOfChannels;
    size_t m_outputFramesReady { 0 };
    Function<void(AudioBus*, size_t framesToProcess)> m_provideInput;
    RefPtr<AudioBus> m_multiChannelBus;
};

} // namespace WebCore

#endif // MultiChannelResampler_h
