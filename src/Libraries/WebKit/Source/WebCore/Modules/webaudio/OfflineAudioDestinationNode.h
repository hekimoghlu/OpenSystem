/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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

#include "AudioDestinationNode.h"
#include <wtf/RefPtr.h>
#include <wtf/Threading.h>

namespace WebCore {

class AudioBuffer;
class AudioBus;
class OfflineAudioContext;
    
class OfflineAudioDestinationNode final : public AudioDestinationNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(OfflineAudioDestinationNode);
public:
    OfflineAudioDestinationNode(OfflineAudioContext&, unsigned numberOfChannels, float sampleRate, RefPtr<AudioBuffer>&& renderTarget);
    ~OfflineAudioDestinationNode();

    OfflineAudioContext& context();
    const OfflineAudioContext& context() const;

    AudioBuffer* renderTarget() const { return m_renderTarget.get(); }
    
    // AudioNode   
    void initialize() final;
    void uninitialize() final;

    // AudioDestinationNode
    void enableInput(const String&) final { }
    void startRendering(CompletionHandler<void(std::optional<Exception>&&)>&&) final;

private:
    enum class RenderResult { Failure, Suspended, Complete };
    RenderResult renderOnAudioThread();
    void notifyOfflineRenderingSuspended();

    bool requiresTailProcessing() const final { return false; }

    unsigned maxChannelCount() const final { return m_numberOfChannels; }

    unsigned m_numberOfChannels;

    // This AudioNode renders into this AudioBuffer.
    RefPtr<AudioBuffer> m_renderTarget;
    
    // Temporary AudioBus for each render quantum.
    RefPtr<AudioBus> m_renderBus;
    
    // Rendering thread.
    RefPtr<Thread> m_renderThread;
    size_t m_framesToProcess;
    size_t m_destinationOffset { 0 };
    bool m_startedRendering { false };
};

} // namespace WebCore
