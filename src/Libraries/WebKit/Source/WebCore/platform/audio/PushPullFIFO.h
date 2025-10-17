/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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

#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class AudioBus;

// PushPullFIFO class is an intermediate audio sample storage between
// WebKit-WebAudio and the renderer. The renderer's hardware callback buffer size
// varies on the platform, but the WebAudio always renders 128 frames (render
// quantum, RQ) thus FIFO is needed to handle the general case.
class PushPullFIFO {
    WTF_MAKE_NONCOPYABLE(PushPullFIFO);

public:
    // Maximum FIFO length. (512 render quanta)
    static constexpr size_t maxFIFOLength { 65536 };

    // |fifoLength| cannot exceed |maxFIFOLength|. Otherwise it crashes.
    WEBCORE_EXPORT PushPullFIFO(unsigned numberOfChannels, size_t fifoLength);
    WEBCORE_EXPORT ~PushPullFIFO();

    // Pushes the rendered frames by WebAudio engine.
    //  - The |inputBus| length is 128 frames (1 render quantum), fixed.
    //  - In case of overflow (FIFO full while push), the existing frames in FIFO
    //    will be overwritten and |indexRead| will be forcibly moved to
    //    |indexWrite| to avoid reading overwritten frames.
    WEBCORE_EXPORT void push(const AudioBus* inputBus);

    // Pulls |framesRequested| by the audio device thread and returns the actual
    // number of frames to be rendered by the source. (i.e. WebAudio graph)
    WEBCORE_EXPORT size_t pull(AudioBus* outputBus, size_t framesRequested);

    size_t framesAvailable() const { return m_framesAvailable; }
    size_t length() const { return m_fifoLength; }
    unsigned numberOfChannels() const;
    AudioBus* bus() const { return m_fifoBus.get(); }

private:
    // The size of the FIFO.
    const size_t m_fifoLength = 0;

    RefPtr<AudioBus> m_fifoBus;

    // The number of frames in the FIFO actually available for pulling.
    size_t m_framesAvailable { 0 };

    size_t m_indexRead { 0 };
    size_t m_indexWrite { 0 };
};

} // namespace WebCore


