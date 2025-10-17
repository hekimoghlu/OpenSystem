/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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

#include "AudioBus.h"
#include <wtf/HashSet.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class AudioNodeOutput;
class BaseAudioContext;
class WeakPtrImplWithEventTargetData;

// An AudioSummingJunction represents a point where zero, one, or more AudioNodeOutputs connect.

class AudioSummingJunction {
public:
    explicit AudioSummingJunction(BaseAudioContext&);
    virtual ~AudioSummingJunction();

    // Can be called from any thread.
    BaseAudioContext* context() { return m_context.get(); }
    const BaseAudioContext* context() const { return m_context.get(); }

    // This copies m_outputs to m_renderingOutputs. Please see comments for these lists below.
    // This must be called when we own the context's graph lock in the audio thread at the very start or end of the render quantum.
    void updateRenderingState();

    // Rendering code accesses its version of the current connections here.
    unsigned numberOfRenderingConnections() const { return m_renderingOutputs.size(); }
    AudioNodeOutput* renderingOutput(unsigned i) { return m_renderingOutputs[i]; }
    const AudioNodeOutput* renderingOutput(unsigned i) const { return m_renderingOutputs[i]; }
    bool isConnected() const { return numberOfRenderingConnections() > 0; }

    virtual bool canUpdateState() = 0;
    virtual void didUpdate() = 0;

    bool addOutput(AudioNodeOutput&);
    bool removeOutput(AudioNodeOutput&);

    void markRenderingStateAsDirty();

    // numberOfConnections() should never be called from the audio rendering thread.
    // Instead numberOfRenderingConnections() and renderingOutput() should be used.
    unsigned numberOfConnections() const { return m_outputs.size(); }

protected:
    WeakPtr<BaseAudioContext, WeakPtrImplWithEventTargetData> m_context;

    unsigned maximumNumberOfChannels() const;

    // m_renderingOutputs is a copy of m_outputs which will never be modified during the graph rendering on the audio thread.
    // This is the list which is used by the rendering code.
    // Whenever m_outputs is modified, the context is told so it can later update m_renderingOutputs from m_outputs at a safe time.
    // Most of the time, m_renderingOutputs is identical to m_outputs.
    Vector<AudioNodeOutput*> m_renderingOutputs;

    // m_renderingStateNeedUpdating keeps track if m_outputs is modified.
    bool m_renderingStateNeedUpdating { false };

private:
    // m_outputs contains the AudioNodeOutputs representing current connections which are not disabled.
    // The rendering code should never use this directly, but instead uses m_renderingOutputs.
    HashSet<AudioNodeOutput*> m_outputs;
    Vector<AudioNodeOutput*> m_pendingRenderingOutputs;
};

} // namespace WebCore
