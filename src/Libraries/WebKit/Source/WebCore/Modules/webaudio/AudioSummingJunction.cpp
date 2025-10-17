/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
#include "config.h"

#if ENABLE(WEB_AUDIO)

#include "AudioSummingJunction.h"

#include "AudioContext.h"
#include "AudioNodeOutput.h"
#include <algorithm>

namespace WebCore {

AudioSummingJunction::AudioSummingJunction(BaseAudioContext& context)
    : m_context(context, EnableWeakPtrThreadingAssertions::No) // WebAudio code uses locking when accessing the context.
{
}

AudioSummingJunction::~AudioSummingJunction()
{
    if (m_renderingStateNeedUpdating && context())
        context()->removeMarkedSummingJunction(this);
}

void AudioSummingJunction::markRenderingStateAsDirty()
{
    ASSERT(context());
    ASSERT(context()->isGraphOwner());
    if (!m_renderingStateNeedUpdating && canUpdateState()) {
        context()->markSummingJunctionDirty(this);
        m_renderingStateNeedUpdating = true;
    }
}

bool AudioSummingJunction::addOutput(AudioNodeOutput& output)
{
    ASSERT(context());
    ASSERT(context()->isGraphOwner());
    if (!m_outputs.add(&output).isNewEntry)
        return false;

    if (m_pendingRenderingOutputs.isEmpty())
        m_pendingRenderingOutputs = copyToVector(m_outputs);
    else
        m_pendingRenderingOutputs.append(&output);

    markRenderingStateAsDirty();
    return true;
}

bool AudioSummingJunction::removeOutput(AudioNodeOutput& output)
{
    ASSERT(context());
    ASSERT(context()->isGraphOwner());

    // Heap allocations are forbidden on the audio thread for performance reasons so we need to
    // explicitly allow the following allocation(s).
    DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;

    if (!m_outputs.remove(&output))
        return false;

    if (m_pendingRenderingOutputs.isEmpty()) {
        m_pendingRenderingOutputs = copyToVector(m_outputs);
    } else
        m_pendingRenderingOutputs.removeFirst(&output);

    markRenderingStateAsDirty();
    return true;
}

void AudioSummingJunction::updateRenderingState()
{
    ASSERT(context());
    ASSERT(context()->isAudioThread() && context()->isGraphOwner());

    if (m_renderingStateNeedUpdating && canUpdateState()) {
        // Copy from m_outputs to m_renderingOutputs.
        m_renderingOutputs = std::exchange(m_pendingRenderingOutputs, { });
        for (auto& output : m_renderingOutputs)
            output->updateRenderingState();

        didUpdate();

        m_renderingStateNeedUpdating = false;
    }
}

unsigned AudioSummingJunction::maximumNumberOfChannels() const
{
    unsigned maxChannels = 0;
    for (auto* output : m_outputs) {
        // Use output()->numberOfChannels() instead of output->bus()->numberOfChannels(),
        // because the calling of AudioNodeOutput::bus() is not safe here.
        maxChannels = std::max(maxChannels, output->numberOfChannels());
    }
    return maxChannels;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
