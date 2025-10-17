/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
#include <wtf/CompletionHandler.h>

namespace WebCore {

class AudioBus;
struct AudioIOPosition;

class AudioDestinationNode : public AudioNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AudioDestinationNode);
public:
    ~AudioDestinationNode();
    
    // AudioNode   
    void process(size_t) final { } // we're pulled by hardware so this is never called

    float sampleRate() const final { return m_sampleRate; }

    size_t currentSampleFrame() const { return m_currentSampleFrame; }
    double currentTime() const { return currentSampleFrame() / static_cast<double>(sampleRate()); }

    virtual unsigned maxChannelCount() const = 0;

    // Enable local/live input for the specified device.
    virtual void enableInput(const String& inputDeviceId) = 0;

    virtual void startRendering(CompletionHandler<void(std::optional<Exception>&&)>&&) = 0;
    virtual void restartRendering() { }

    // AudioDestinationNodes are owned by the BaseAudioContext so we forward the refcounting to its BaseAudioContext.
    void ref() const final;
    void deref() const final;

protected:
    AudioDestinationNode(BaseAudioContext&, float sampleRate);

    double tailTime() const final { return 0; }
    double latencyTime() const final { return 0; }

    void renderQuantum(AudioBus* destinationBus, size_t numberOfFrames, const AudioIOPosition& outputPosition);

private:
    // Counts the number of sample-frames processed by the destination.
    std::atomic<size_t> m_currentSampleFrame { 0 };
    float m_sampleRate;
};

} // namespace WebCore
