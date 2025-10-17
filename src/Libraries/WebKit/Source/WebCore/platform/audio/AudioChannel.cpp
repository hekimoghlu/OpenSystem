/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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

#include "AudioChannel.h"

#include "VectorMath.h"
#include <algorithm>
#include <math.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioChannel);

void AudioChannel::scale(float scale)
{
    if (isSilent())
        return;

    VectorMath::multiplyByScalar(span(), scale, mutableSpan());
}

void AudioChannel::copyFrom(const AudioChannel* sourceChannel)
{
    bool isSafe = (sourceChannel && sourceChannel->length() >= length());
    ASSERT(isSafe);

    if (!isSafe || sourceChannel->isSilent()) {
        zero();
        return;
    }
    memcpySpan(mutableSpan(), sourceChannel->span().first(length()));
}

void AudioChannel::copyFromRange(const AudioChannel* sourceChannel, unsigned startFrame, unsigned endFrame)
{
    // Check that range is safe for reading from sourceChannel.
    bool isRangeSafe = sourceChannel && startFrame < endFrame && endFrame <= sourceChannel->length();
    ASSERT(isRangeSafe);
    if (!isRangeSafe)
        return;

    if (sourceChannel->isSilent() && isSilent())
        return;

    // Check that this channel has enough space.
    size_t rangeLength = endFrame - startFrame;
    bool isRangeLengthSafe = rangeLength <= length();
    ASSERT(isRangeLengthSafe);
    if (!isRangeLengthSafe)
        return;

    auto source = sourceChannel->span();
    auto destination = mutableSpan();

    if (sourceChannel->isSilent()) {
        if (rangeLength == length())
            zero();
        else
            zeroSpan(destination.first(rangeLength));
    } else
        memcpySpan(destination, source.subspan(startFrame, rangeLength));
}

void AudioChannel::sumFrom(const AudioChannel* sourceChannel)
{
    bool isSafe = sourceChannel && sourceChannel->length() >= length();
    ASSERT(isSafe);
    if (!isSafe)
        return;

    if (sourceChannel->isSilent())
        return;

    if (isSilent())
        copyFrom(sourceChannel);
    else
        VectorMath::add(span(), sourceChannel->span().first(length()), mutableSpan());
}

float AudioChannel::maxAbsValue() const
{
    if (isSilent())
        return 0;

    return VectorMath::maximumMagnitude(span());
}

} // WebCore

#endif // ENABLE(WEB_AUDIO)
