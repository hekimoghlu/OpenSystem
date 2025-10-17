/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 27, 2023.
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

#if ENABLE(VIDEO)

#include "ExceptionOr.h"
#include "HTMLMediaElementEnums.h"

namespace WebCore {

class TimeRanges;

class MediaControllerInterface : public HTMLMediaElementEnums {
public:
    virtual ~MediaControllerInterface() { };
    
    // MediaController IDL:
    virtual Ref<TimeRanges> buffered() const = 0;
    virtual Ref<TimeRanges> seekable() const = 0;
    virtual Ref<TimeRanges> played() = 0;
    
    virtual double duration() const = 0;
    virtual double currentTime() const = 0;
    virtual void setCurrentTime(double) = 0;
    
    virtual bool paused() const = 0;
    virtual void play() = 0;
    virtual void pause() = 0;
    
    virtual double defaultPlaybackRate() const = 0;
    virtual void setDefaultPlaybackRate(double) = 0;
    
    virtual double playbackRate() const = 0;
    virtual void setPlaybackRate(double) = 0;
    
    virtual double volume() const = 0;
    virtual ExceptionOr<void> setVolume(double) = 0;
    
    virtual bool muted() const = 0;
    virtual void setMuted(bool) = 0;

    using HTMLMediaElementEnums::ReadyState;
    virtual ReadyState readyState() const = 0;

    // MediaControlElements:
    virtual bool supportsFullscreen(HTMLMediaElementEnums::VideoFullscreenMode) const = 0;
    virtual bool isFullscreen() const = 0;
    virtual void enterFullscreen() = 0;

    virtual bool hasAudio() const = 0;
    virtual bool hasVideo() const = 0;
    virtual bool hasClosedCaptions() const = 0;
    virtual void setClosedCaptionsVisible(bool) = 0;
    virtual bool closedCaptionsVisible() const = 0;

    virtual bool supportsScanning() const = 0;

    virtual void beginScrubbing() = 0;
    virtual void endScrubbing() = 0;

    enum ScanDirection { Backward, Forward };
    virtual void beginScanning(ScanDirection) = 0;
    virtual void endScanning() = 0;

    virtual bool canPlay() const = 0;

    virtual bool isLiveStream() const = 0;

    virtual bool hasCurrentSrc() const = 0;

    virtual void returnToRealtime() = 0;
};

}

#endif
