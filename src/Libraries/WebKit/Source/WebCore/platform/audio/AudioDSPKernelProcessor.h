/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
#ifndef AudioDSPKernelProcessor_h
#define AudioDSPKernelProcessor_h

#include "AudioBus.h"
#include "AudioProcessor.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class AudioBus;
class AudioDSPKernel;
class AudioProcessor;

// AudioDSPKernelProcessor processes one input -> one output (N channels each)
// It uses one AudioDSPKernel object per channel to do the processing, thus there is no cross-channel processing.
// Despite this limitation it turns out to be a very common and useful type of processor.

class AudioDSPKernelProcessor : public AudioProcessor {
    WTF_MAKE_TZONE_ALLOCATED(AudioDSPKernelProcessor);
public:
    // numberOfChannels may be later changed if object is not yet in an "initialized" state
    AudioDSPKernelProcessor(float sampleRate, unsigned numberOfChannels);
    virtual ~AudioDSPKernelProcessor();

    // Subclasses create the appropriate type of processing kernel here.
    // We'll call this to create a kernel for each channel.
    virtual std::unique_ptr<AudioDSPKernel> createKernel() = 0;

    // AudioProcessor methods
    void initialize() override;
    void uninitialize() override;
    void process(const AudioBus* source, AudioBus* destination, size_t framesToProcess) override;
    void processOnlyAudioParams(size_t framesToProcess) override;
    void reset() override;
    void setNumberOfChannels(unsigned) override;
    unsigned numberOfChannels() const override { return m_numberOfChannels; }

    double tailTime() const override;
    double latencyTime() const override;
    bool requiresTailProcessing() const override;

protected:
    Vector<std::unique_ptr<AudioDSPKernel>> m_kernels;
    bool m_hasJustReset { true };
};

} // namespace WebCore

#endif // AudioDSPKernelProcessor_h
