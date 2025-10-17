/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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

#include "ActiveDOMObject.h"
#include "AudioBus.h"
#include "AudioNode.h"
#include "EventListener.h"
#include "EventTarget.h"
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

class AudioBuffer;
class AudioContext;
class AudioProcessingEvent;

// ScriptProcessorNode is an AudioNode which allows for arbitrary synthesis or processing directly using JavaScript.
// The API allows for a variable number of inputs and outputs, although it must have at least one input or output.
// This basic implementation supports no more than one input and output.
// The "onaudioprocess" attribute is an event listener which will get called periodically with an AudioProcessingEvent which has
// AudioBuffers for each input and output.

class ScriptProcessorNode final : public AudioNode, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ScriptProcessorNode);
public:
    // bufferSize must be one of the following values: 256, 512, 1024, 2048, 4096, 8192, 16384.
    // This value controls how frequently the onaudioprocess event handler is called and how many sample-frames need to be processed each call.
    // Lower numbers for bufferSize will result in a lower (better) latency. Higher numbers will be necessary to avoid audio breakup and glitches.
    // The value chosen must carefully balance between latency and audio quality.
    static Ref<ScriptProcessorNode> create(BaseAudioContext&, size_t bufferSize, unsigned numberOfInputChannels, unsigned numberOfOutputChannels);

    virtual ~ScriptProcessorNode();

    // ActiveDOMObject.
    void ref() const final { AudioNode::ref(); }
    void deref() const final { AudioNode::deref(); }

    // AudioNode
    void process(size_t framesToProcess) override;
    void initialize() override;
    void uninitialize() override;

    size_t bufferSize() const { return m_bufferSize; }

    ExceptionOr<void> setChannelCount(unsigned) final;
    ExceptionOr<void> setChannelCountMode(ChannelCountMode) final;

private:
    double tailTime() const override;
    double latencyTime() const override;
    bool requiresTailProcessing() const final;

    ScriptProcessorNode(BaseAudioContext&, size_t bufferSize, unsigned numberOfInputChannels, unsigned numberOfOutputChannels);

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    void eventListenersDidChange() final;
    void fireProcessEvent(unsigned bufferIndex);

    RefPtr<AudioBuffer> createInputBufferForJS(AudioBuffer*) const;
    RefPtr<AudioBuffer> createOutputBufferForJS(AudioBuffer&) const;

    // Double buffering.
    static constexpr unsigned bufferCount = 2;
    unsigned bufferIndex() const { return m_bufferIndex; }
    void swapBuffers() { m_bufferIndex = (m_bufferIndex + 1) % bufferCount; }

    unsigned m_bufferIndex { 0 };
    std::array<Lock, bufferCount> m_bufferLocks;
    std::array<RefPtr<AudioBuffer>, bufferCount> m_inputBuffers;
    std::array<RefPtr<AudioBuffer>, bufferCount> m_outputBuffers;
    mutable RefPtr<AudioBuffer> m_cachedInputBufferForJS;
    mutable RefPtr<AudioBuffer> m_cachedOutputBufferForJS;

    size_t m_bufferSize;
    unsigned m_bufferReadWriteIndex { 0 };

    unsigned m_numberOfInputChannels;
    unsigned m_numberOfOutputChannels;

    RefPtr<AudioBus> m_internalInputBus;
    bool m_hasAudioProcessEventListener { false };
};

} // namespace WebCore
