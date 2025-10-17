/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#include "AudioNode.h"

namespace WebCore {

class AudioScheduledSourceNode : public AudioNode, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AudioScheduledSourceNode);
public:
    // ActiveDOMObject.
    void ref() const final { AudioNode::ref(); }
    void deref() const final { AudioNode::deref(); }

    // These are the possible states an AudioScheduledSourceNode can be in:
    //
    // UNSCHEDULED_STATE - Initial playback state. Created, but not yet scheduled.
    // SCHEDULED_STATE - Scheduled to play but not yet playing.
    // PLAYING_STATE - Generating sound.
    // FINISHED_STATE - Finished generating sound.
    //
    // The state can only transition to the next state, except for the FINISHED_STATE which can
    // never be changed.
    enum PlaybackState {
        // These must be defined with the same names and values as in the .idl file.
        UNSCHEDULED_STATE = 0,
        SCHEDULED_STATE = 1,
        PLAYING_STATE = 2,
        FINISHED_STATE = 3
    };

    ExceptionOr<void> startLater(double when);
    ExceptionOr<void> stopLater(double when);

    unsigned short playbackState() const { return static_cast<unsigned short>(m_playbackState); }
    bool isPlayingOrScheduled() const { return m_playbackState == PLAYING_STATE || m_playbackState == SCHEDULED_STATE; }
    bool hasFinished() const { return m_playbackState == FINISHED_STATE; }

protected:
    AudioScheduledSourceNode(BaseAudioContext&, NodeType);

    // Get frame information for the current time quantum.
    // We handle the transition into PLAYING_STATE and FINISHED_STATE here,
    // zeroing out portions of the outputBus which are outside the range of startFrame and endFrame.
    // Each frame time is relative to the context's currentSampleFrame().
    // quantumFrameOffset: Offset frame in this time quantum to start rendering.
    // nonSilentFramesToProcess: Number of frames rendering non-silence (will be <= quantumFrameSize).
    // startFrameOffset : The fractional frame offset from quantumFrameOffset and the actual starting
    //                    time of the source. This is non-zero only when transitioning from the
    //                    SCHEDULED_STATE to the PLAYING_STATE.
    void updateSchedulingInfo(size_t quantumFrameSize, AudioBus& outputBus, size_t& quantumFrameOffset, size_t& nonSilentFramesToProcess, double& startFrameOffset);

    // Called when we have no more sound to play or the noteOff() time has been reached.
    virtual void finish();

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    void eventListenersDidChange() final;

    bool requiresTailProcessing() const final { return false; }

    // This is accessed from the main thread and the audio thread.
    std::atomic<PlaybackState> m_playbackState { UNSCHEDULED_STATE };

    // m_startTime is the time to start playing based on the context's timeline (0 or a time less than the context's current time means "now").
    double m_startTime { 0 }; // in seconds

    // m_endTime is the time to stop playing based on the context's timeline (0 or a time less than the context's current time means "now").
    // If it hasn't been set explicitly, then the sound will not stop playing (if looping) or will stop when the end of the AudioBuffer
    // has been reached.
    std::optional<double> m_endTime; // in seconds
    bool m_hasEndedEventListener { false };
};

} // namespace WebCore
