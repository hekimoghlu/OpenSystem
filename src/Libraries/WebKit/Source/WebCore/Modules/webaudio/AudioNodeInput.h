/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
#include "AudioNode.h"
#include "AudioSummingJunction.h"
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class AudioNode;
class AudioNodeOutput;

// An AudioNodeInput represents an input to an AudioNode and can be connected from one or more AudioNodeOutputs.
// In the case of multiple connections, the input will act as a unity-gain summing junction, mixing all the outputs.
// The number of channels of the input's bus is the maximum of the number of channels of all its connections.

class AudioNodeInput final : public AudioSummingJunction {
    WTF_MAKE_NONCOPYABLE(AudioNodeInput);
    WTF_MAKE_TZONE_ALLOCATED(AudioNodeInput);
public:
    explicit AudioNodeInput(AudioNode*);

    // AudioSummingJunction
    bool canUpdateState() override { return !node()->isMarkedForDeletion(); }
    void didUpdate() override;

    // Can be called from any thread.
    AudioNode* node() const { return m_node.get(); }

    // Must be called with the context's graph lock.
    void connect(AudioNodeOutput*);
    void disconnect(AudioNodeOutput*);

    // disable() will take the output out of the active connections list and set aside in a disabled list.
    // enable() will put the output back into the active connections list.
    // Must be called with the context's graph lock.
    void enable(AudioNodeOutput*);
    void disable(AudioNodeOutput*);

    // pull() processes all of the AudioNodes connected to us.
    // In the case of multiple connections it sums the result into an internal summing bus.
    // In the single connection case, it allows in-place processing where possible using inPlaceBus.
    // It returns the bus which it rendered into, returning inPlaceBus if in-place processing was performed.
    // Called from context's audio thread.
    AudioBus* pull(AudioBus* inPlaceBus, size_t framesToProcess);

    // bus() contains the rendered audio after pull() has been called for each time quantum.
    // Called from context's audio thread.
    AudioBus* bus();
    
    // updateInternalBus() updates m_internalSummingBus appropriately for the number of channels.
    // This must be called when we own the context's graph lock in the audio thread at the very start or end of the render quantum.
    void updateInternalBus();

    // The number of channels of the connection with the largest number of channels.
    unsigned numberOfChannels() const;        
    
private:
    WeakPtr<AudioNode, WeakPtrImplWithEventTargetData> m_node;

    // m_disabledOutputs contains the AudioNodeOutputs which are disabled (will not be processed) by the audio graph rendering.
    // But, from JavaScript's perspective, these outputs are still connected to us.
    // Generally, these represent disabled connections from "notes" which have finished playing but are not yet garbage collected.
    HashSet<AudioNodeOutput*> m_disabledOutputs;

    // Called from context's audio thread.
    AudioBus* internalSummingBus();
    void sumAllConnections(AudioBus* summingBus, size_t framesToProcess);

    RefPtr<AudioBus> m_internalSummingBus;
};

} // namespace WebCore
