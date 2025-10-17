/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#import "config.h"
#import "WKVelocityTrackingScrollView.h"

#if PLATFORM(IOS_FAMILY)

#import <wtf/ApproximateTime.h>

template <size_t windowSize>
struct ScrollingDeltaWindow {
public:
    static constexpr auto maxDeltaDuration = 100_ms;

    void update(CGPoint contentOffset)
    {
        auto currentTime = ApproximateTime::now();
        auto deltaDuration = currentTime - m_lastTimestamp;
        if (deltaDuration > maxDeltaDuration)
            reset();
        else {
            m_deltas[m_lastIndex] = {
                CGSizeMake(contentOffset.x - m_lastOffset.x, contentOffset.y - m_lastOffset.y),
                deltaDuration
            };
            m_lastIndex = ++m_lastIndex % windowSize;
        }
        m_lastTimestamp = currentTime;
        m_lastOffset = contentOffset;
    }

    void reset()
    {
        for (auto& delta : m_deltas)
            delta = { CGSizeZero, 0_ms };
    }

    CGSize averageVelocity() const
    {
        if (ApproximateTime::now() - m_lastTimestamp > maxDeltaDuration)
            return CGSizeZero;

        auto cumulativeDelta = CGSizeZero;
        CGFloat numberOfDeltas = 0;
        for (auto [delta, duration] : m_deltas) {
            if (!duration)
                continue;

            cumulativeDelta.width += delta.width / duration.seconds();
            cumulativeDelta.height += delta.height / duration.seconds();
            numberOfDeltas += 1;
        }

        if (!numberOfDeltas)
            return CGSizeZero;

        cumulativeDelta.width /= numberOfDeltas;
        cumulativeDelta.height /= numberOfDeltas;
        return cumulativeDelta;
    }

private:
    std::array<std::pair<CGSize, Seconds>, windowSize> m_deltas;
    size_t m_lastIndex { 0 };
    ApproximateTime m_lastTimestamp;
    CGPoint m_lastOffset { CGPointZero };
};

@implementation WKVelocityTrackingScrollView {
    ScrollingDeltaWindow<3> _scrollingDeltaWindow;
}

- (void)updateInteractiveScrollVelocity
{
    if (!self.tracking && !self.decelerating)
        return;

    _scrollingDeltaWindow.update(self.contentOffset);
}

- (CGSize)interactiveScrollVelocityInPointsPerSecond
{
    return _scrollingDeltaWindow.averageVelocity();
}

@end

#endif // PLATFORM(IOS_FAMILY)
