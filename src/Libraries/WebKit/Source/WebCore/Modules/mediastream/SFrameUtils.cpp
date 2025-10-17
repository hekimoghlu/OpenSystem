/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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
#include "SFrameUtils.h"

#if ENABLE(WEB_RTC)

#include <wtf/Function.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

static inline bool isSliceNALU(uint8_t data)
{
    return (data & 0x1F) == 1;
}

static inline bool isSPSNALU(uint8_t data)
{
    return (data & 0x1F) == 7;
}

static inline bool isPPSNALU(uint8_t data)
{
    return (data & 0x1F) == 8;
}

static inline bool isIDRNALU(uint8_t data)
{
    return (data & 0x1F) == 5;
}

static inline void findNalus(std::span<const uint8_t> frameData, size_t offset, const Function<bool(size_t)>& callback)
{
    for (size_t i = 4 + offset; i < frameData.size(); ++i) {
        if (frameData[i - 4] == 0 && frameData[i - 3] == 0 && frameData[i - 2] == 0 && frameData[i - 1] == 1) {
            if (callback(i))
                return;
        }
    }
}

size_t computeH264PrefixOffset(std::span<const uint8_t> frameData)
{
    size_t offset = 0;
    findNalus(frameData, 0, [&offset, frameData](auto position) {
        if (isIDRNALU(frameData[position]) || isSliceNALU(frameData[position])) {
            // Skip 00 00 00 01, nalu type byte and the next byte.
            offset = position + 2;
            return true;
        }
        return false;
    });
    return offset;
}

bool needsRbspUnescaping(std::span<const uint8_t> frameData)
{
    for (size_t i = 0; i < frameData.size() - 3; ++i) {
        if (frameData[i] == 0 && frameData[i + 1] == 0 && frameData[i + 2] == 3)
            return true;
    }
    return false;
}

Vector<uint8_t> fromRbsp(std::span<const uint8_t> frameData)
{
    Vector<uint8_t> buffer;
    buffer.reserveInitialCapacity(frameData.size());

    size_t i;
    for (i = 0; i < frameData.size() - 3; ++i) {
        if (frameData[i] == 0 && frameData[i + 1] == 0 && frameData[i + 2] == 3) {
            buffer.append(frameData[i]);
            buffer.append(frameData[i + 1]);
            // Skip next byte which is delimiter.
            i += 2;
        } else
            buffer.append(frameData[i]);
    }
    for (; i < frameData.size(); ++i)
        buffer.append(frameData[i]);

    return buffer;
}

SFrameCompatibilityPrefixBuffer computeH264PrefixBuffer(std::span<const uint8_t> frameData)
{
    // Delta and key prefixes assume SPS/PPS with IDs equal to 0 have been transmitted.
    static const uint8_t prefixDeltaFrame[6] = { 0x00, 0x00, 0x00, 0x01, 0x21, 0xe0 };

    if (frameData.size() < 5)
        return std::span<const uint8_t> { };

    // We assume a key frame starts with SPS, then PPS. Otherwise we wrap it as a delta frame.
    if (!isSPSNALU(frameData[4]))
        return std::span<const uint8_t> { prefixDeltaFrame };

    // Search for PPS
    size_t spsPpsLength = 0;
    findNalus(frameData, 5, [frameData, &spsPpsLength](auto position) {
        if (isPPSNALU(frameData[position]))
            spsPpsLength = position;
        return true;
    });
    if (!spsPpsLength)
        return std::span<const uint8_t> { prefixDeltaFrame };

    // Search for next NALU to compute the real spsPpsLength, including the next 00 00 00 01.
    findNalus(frameData, spsPpsLength + 1, [&spsPpsLength](auto position) {
        spsPpsLength = position;
        return true;
    });

    Vector<uint8_t> buffer(spsPpsLength + 2);
IGNORE_GCC_WARNINGS_BEGIN("restrict")
    // https://bugs.webkit.org/show_bug.cgi?id=246862
    memcpySpan(buffer.mutableSpan(), frameData.first(spsPpsLength));
IGNORE_GCC_WARNINGS_END
    buffer[spsPpsLength] = 0x25;
    buffer[spsPpsLength + 1] = 0xb8;
    return buffer;
}

static inline void findEscapeRbspPatterns(const Vector<uint8_t>& frame, size_t offset, const Function<void(size_t, bool)>& callback)
{
    size_t numConsecutiveZeros = 0;
    auto data = frame.span();
    for (size_t i = offset; i < frame.size(); ++i) {
        bool shouldEscape = data[i] <= 3 && numConsecutiveZeros >= 2;
        if (shouldEscape)
            numConsecutiveZeros = 0;

        if (data[i] == 0)
            ++numConsecutiveZeros;
        else
            numConsecutiveZeros = 0;

        callback(i, shouldEscape);
    }
}

void toRbsp(Vector<uint8_t>& frame, size_t offset)
{
    size_t count = 0;
    findEscapeRbspPatterns(frame, offset, [&count](size_t, bool shouldBeEscaped) {
        if (shouldBeEscaped)
            ++count;
    });
    if (!count)
        return;

    Vector<uint8_t> newFrame;
    newFrame.reserveInitialCapacity(frame.size() + count);
    newFrame.append(frame.subspan(0, offset));

    findEscapeRbspPatterns(frame, offset, [data = frame.span(), &newFrame](size_t position, bool shouldBeEscaped) {
        if (shouldBeEscaped)
            newFrame.append(3);
        newFrame.append(data[position]);
    });

    frame = WTFMove(newFrame);
}

static inline bool isVP8KeyFrame(std::span<const uint8_t> frame)
{
    ASSERT(frame.size());
    return !(frame.front() & 0x01);
}

size_t computeVP8PrefixOffset(std::span<const uint8_t> frame)
{
    return isVP8KeyFrame(frame) ? 10 : 3;
}

SFrameCompatibilityPrefixBuffer computeVP8PrefixBuffer(std::span<const uint8_t> frame)
{
    Vector<uint8_t> prefix(frame.first(isVP8KeyFrame(frame) ? 10 : 3));
    return prefix;
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
