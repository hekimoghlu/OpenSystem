/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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

#include <wtf/MonotonicTime.h>
#include <wtf/Seconds.h>

namespace WebCore {

class AudioBus;

struct AudioIOPosition {
    // Audio stream position in seconds.
    Seconds position;
    // System's monotonic time in seconds corresponding to the contained |position| value.
    MonotonicTime timestamp;
};

// Abstract base-class for isochronous audio I/O client.
class AudioIOCallback {
public:
    // render() is called periodically to get the next render quantum of audio into destinationBus.
    // Optional audio input is given in sourceBus (if it's not 0).
    virtual void render(AudioBus* sourceBus, AudioBus* destinationBus, size_t framesToProcess, const AudioIOPosition& outputPosition) = 0;

    virtual void isPlayingDidChange() = 0;

    virtual ~AudioIOCallback() = default;
};

} // WebCore
