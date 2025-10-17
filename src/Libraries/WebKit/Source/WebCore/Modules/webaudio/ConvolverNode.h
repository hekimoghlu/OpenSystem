/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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

#include "AudioNode.h"
#include "ConvolverOptions.h"
#include <wtf/Lock.h>

namespace WebCore {

class AudioBuffer;
class Reverb;
    
class ConvolverNode final : public AudioNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ConvolverNode);
public:
    static ExceptionOr<Ref<ConvolverNode>> create(BaseAudioContext&, ConvolverOptions&& = { });
    
    virtual ~ConvolverNode();
    
    ExceptionOr<void> setBufferForBindings(RefPtr<AudioBuffer>&&);
    AudioBuffer* bufferForBindings(); // Only safe to call on the main thread.

    bool normalizeForBindings() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_normalize; }
    void setNormalizeForBindings(bool);

    ExceptionOr<void> setChannelCount(unsigned) final;
    ExceptionOr<void> setChannelCountMode(ChannelCountMode) final;

private:
    explicit ConvolverNode(BaseAudioContext&);

    double tailTime() const final;
    double latencyTime() const final;
    bool requiresTailProcessing() const final;

    void process(size_t framesToProcess) final;
    void checkNumberOfChannelsForInput(AudioNodeInput*) final;

    std::unique_ptr<Reverb> m_reverb WTF_GUARDED_BY_LOCK(m_processLock); // Only modified on the main thread but accessed on the audio thread.
    RefPtr<AudioBuffer> m_buffer WTF_GUARDED_BY_LOCK(m_processLock); // Only modified on the main thread but accessed on the audio thread.

    // This synchronizes dynamic changes to the convolution impulse response with process().
    mutable Lock m_processLock;

    // Normalize the impulse response or not.
    bool m_normalize { true }; // Only used on the main thread.
};

} // namespace WebCore
