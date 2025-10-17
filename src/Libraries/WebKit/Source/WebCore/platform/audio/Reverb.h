/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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
#ifndef Reverb_h
#define Reverb_h

#include "ReverbConvolver.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class AudioBus;
    
// Multi-channel convolution reverb with channel matrixing - one or more ReverbConvolver objects are used internally.

class Reverb final {
    WTF_MAKE_TZONE_ALLOCATED(Reverb);
public:
    enum { MaxFrameSize = 256 };

    // renderSliceSize is a rendering hint, so the FFTs can be optimized to not all occur at the same time (very bad when rendering on a real-time thread).
    Reverb(AudioBus* impulseResponseBuffer, size_t renderSliceSize, size_t maxFFTSize, bool useBackgroundThreads, bool normalize);

    void process(const AudioBus* sourceBus, AudioBus* destinationBus, size_t framesToProcess);
    void reset();

    size_t impulseResponseLength() const { return m_impulseResponseLength; }
    size_t latencyFrames() const;

private:
    void initialize(AudioBus* impulseResponseBuffer, size_t renderSliceSize, size_t maxFFTSize, bool useBackgroundThreads, float scale);

    size_t m_impulseResponseLength;

    // The actual number of channels in the response. This can be less
    // than the number of ReverbConvolver's in |m_convolvers|.
    unsigned m_numberOfResponseChannels { 0 };
    Vector<std::unique_ptr<ReverbConvolver>> m_convolvers;

    // For "True" stereo processing
    RefPtr<AudioBus> m_tempBuffer;
};

} // namespace WebCore

#endif // Reverb_h
