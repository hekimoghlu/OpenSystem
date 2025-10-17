/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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

#include "AudioScheduledSourceNode.h"

#include "AudioContext.h"
#include "AudioUtilities.h"
#include "Event.h"
#include "EventNames.h"
#include "ScriptController.h"
#include "ScriptExecutionContext.h"
#include <algorithm>
#include <wtf/MathExtras.h>
#include <wtf/Scope.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(IOS_FAMILY)
#include "ScriptController.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AudioScheduledSourceNode);

AudioScheduledSourceNode::AudioScheduledSourceNode(BaseAudioContext& context, NodeType type)
    : AudioNode(context, type)
    , ActiveDOMObject(context.scriptExecutionContext())
{
}

void AudioScheduledSourceNode::updateSchedulingInfo(size_t quantumFrameSize, AudioBus& outputBus, size_t& quantumFrameOffset, size_t& nonSilentFramesToProcess, double& startFrameOffset)
{
    nonSilentFramesToProcess = 0;
    quantumFrameOffset = 0;
    startFrameOffset = 0;

    ASSERT(quantumFrameSize == AudioUtilities::renderQuantumSize);
    if (quantumFrameSize != AudioUtilities::renderQuantumSize)
        return;

    double sampleRate = this->sampleRate();

    // quantumStartFrame     : Start frame of the current time quantum.
    // quantumEndFrame       : End frame of the current time quantum.
    // startFrame            : Start frame for this source.
    // endFrame              : End frame for this source.
    size_t quantumStartFrame = context().currentSampleFrame();
    size_t quantumEndFrame = quantumStartFrame + quantumFrameSize;

    // Round up if the start time isn't on a frame boundary so we don't start too early.
    size_t startFrame = AudioUtilities::timeToSampleFrame(m_startTime, sampleRate, AudioUtilities::SampleFrameRounding::Up);
    size_t endFrame = 0;
    if (m_endTime) {
        // The end frame is the end time rounded up because it is an exclusive upper
        // bound of the end time. We also need to take care to handle huge end
        // times and clamp the corresponding frame to the largest size_t value.
        endFrame = AudioUtilities::timeToSampleFrame(*m_endTime, sampleRate, AudioUtilities::SampleFrameRounding::Up);
    }

    // If we know the end time and it's already passed, then don't bother doing any more rendering this cycle.
    if (m_endTime && endFrame <= quantumStartFrame)
        finish();

    if (m_playbackState == UNSCHEDULED_STATE || m_playbackState == FINISHED_STATE || startFrame >= quantumEndFrame) {
        // Output silence.
        outputBus.zero();
        return;
    }

    // Check if it's time to start playing.
    if (m_playbackState == SCHEDULED_STATE) {
        // Increment the active source count only if we're transitioning from SCHEDULED_STATE to PLAYING_STATE.
        m_playbackState = PLAYING_STATE;
        // NOTE: startFrameOffset is usually negative, but may not be because of
        // the rounding that may happen in computing |startFrame| above.
        startFrameOffset = m_startTime * sampleRate - startFrame;
    }

    quantumFrameOffset = startFrame > quantumStartFrame ? startFrame - quantumStartFrame : 0;
    quantumFrameOffset = std::min(quantumFrameOffset, quantumFrameSize); // clamp to valid range
    nonSilentFramesToProcess = quantumFrameSize - quantumFrameOffset;

    if (!nonSilentFramesToProcess) {
        // Output silence.
        outputBus.zero();
        return;
    }

    // Handle silence before we start playing.
    // Zero any initial frames representing silence leading up to a rendering start time in the middle of the quantum.
    if (quantumFrameOffset) {
        for (unsigned i = 0; i < outputBus.numberOfChannels(); ++i)
            zeroSpan(outputBus.channel(i)->mutableSpan().first(quantumFrameOffset));
    }

    // Handle silence after we're done playing.
    // If the end time is somewhere in the middle of this time quantum, then zero out the
    // frames from the end time to the very end of the quantum.
    if (m_endTime && endFrame >= quantumStartFrame && endFrame < quantumEndFrame) {
        size_t zeroStartFrame = endFrame - quantumStartFrame;
        size_t framesToZero = quantumFrameSize - zeroStartFrame;

        ASSERT(zeroStartFrame < quantumFrameSize);
        ASSERT(framesToZero <= quantumFrameSize);
        ASSERT(zeroStartFrame + framesToZero <= quantumFrameSize);

        bool isSafe = zeroStartFrame < quantumFrameSize && framesToZero <= quantumFrameSize && zeroStartFrame + framesToZero <= quantumFrameSize;
        if (isSafe) {
            if (framesToZero > nonSilentFramesToProcess)
                nonSilentFramesToProcess = 0;
            else
                nonSilentFramesToProcess -= framesToZero;

            for (unsigned i = 0; i < outputBus.numberOfChannels(); ++i)
                zeroSpan(outputBus.channel(i)->mutableSpan().subspan(zeroStartFrame, framesToZero));
        }

        finish();
    }
}

ExceptionOr<void> AudioScheduledSourceNode::startLater(double when)
{
    ASSERT(isMainThread());
    ALWAYS_LOG(LOGIDENTIFIER, when);

    if (m_playbackState != UNSCHEDULED_STATE)
        return Exception { ExceptionCode::InvalidStateError, "Cannot call start() more than once"_s };

    if (!std::isfinite(when) || when < 0)
        return Exception { ExceptionCode::RangeError, "when value should be positive"_s };

    context().sourceNodeWillBeginPlayback(*this);

    m_startTime = when;
    m_playbackState = SCHEDULED_STATE;

    return { };
}

ExceptionOr<void> AudioScheduledSourceNode::stopLater(double when)
{
    ASSERT(isMainThread());
    ALWAYS_LOG(LOGIDENTIFIER, when);

    if (m_playbackState == UNSCHEDULED_STATE)
        return Exception { ExceptionCode::InvalidStateError, "cannot call stop without calling start first."_s };

    if (!std::isfinite(when) || when < 0)
        return Exception { ExceptionCode::RangeError, "when value should be positive"_s };

    m_endTime = when;

    return { };
}

bool AudioScheduledSourceNode::virtualHasPendingActivity() const
{
    return m_hasEndedEventListener && m_playbackState != FINISHED_STATE && !isMarkedForDeletion() && !context().isClosed();
}

void AudioScheduledSourceNode::eventListenersDidChange()
{
    m_hasEndedEventListener = hasEventListeners(eventNames().endedEvent);
}

void AudioScheduledSourceNode::finish()
{
    ASSERT(!hasFinished());
    // Let the context dereference this AudioNode.
    context().sourceNodeDidFinishPlayback(*this);
    m_playbackState = FINISHED_STATE;

    // Heap allocations are forbidden on the audio thread for performance reasons so we need to
    // explicitly allow the following allocation(s).
    DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;
    callOnMainThread([this, protectedThis = Ref { *this }, pendingActivity = makePendingActivity(*this)] {
        if (context().isStopped())
            return;
        this->dispatchEvent(Event::create(eventNames().endedEvent, Event::CanBubble::No, Event::IsCancelable::No));
    });
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
