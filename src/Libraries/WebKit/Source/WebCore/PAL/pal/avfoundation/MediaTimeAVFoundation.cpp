/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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
#include "MediaTimeAVFoundation.h"

#if USE(AVFOUNDATION)

#include "CoreMediaSoftLink.h"

namespace PAL {

static bool CMTimeHasFlags(const CMTime& cmTime, uint32_t flags)
{
    return (cmTime.flags & flags) == flags;
}

MediaTime toMediaTime(const CMTime& cmTime)
{
    uint32_t flags = 0;
    if (CMTimeHasFlags(cmTime, kCMTimeFlags_Valid))
        flags |= MediaTime::Valid;
    if (CMTimeHasFlags(cmTime, kCMTimeFlags_Valid | kCMTimeFlags_HasBeenRounded))
        flags |= MediaTime::HasBeenRounded;
    if (CMTimeHasFlags(cmTime, kCMTimeFlags_Valid | kCMTimeFlags_PositiveInfinity))
        flags |= MediaTime::PositiveInfinite;
    if (CMTimeHasFlags(cmTime, kCMTimeFlags_Valid | kCMTimeFlags_NegativeInfinity))
        flags |= MediaTime::NegativeInfinite;
    if (CMTimeHasFlags(cmTime, kCMTimeFlags_Valid | kCMTimeFlags_Indefinite))
        flags |= MediaTime::Indefinite;

    return MediaTime(cmTime.value, cmTime.timescale, flags);
}

CMTime toCMTime(const MediaTime& mediaTime)
{
    CMTime time;

    if (mediaTime.hasDoubleValue())
        time = CMTimeMakeWithSeconds(mediaTime.toDouble(), mediaTime.timeScale());
    else
        time = CMTimeMake(mediaTime.timeValue(), mediaTime.timeScale());

    if (mediaTime.isValid())
        time.flags |= kCMTimeFlags_Valid;
    else
        time.flags &= ~kCMTimeFlags_Valid;
    if (mediaTime.hasBeenRounded())
        time.flags |= kCMTimeFlags_HasBeenRounded;
    if (mediaTime.isPositiveInfinite())
        time.flags |= kCMTimeFlags_PositiveInfinity;
    if (mediaTime.isNegativeInfinite())
        time.flags |= kCMTimeFlags_NegativeInfinity;

    return time;
}

}

#endif
