/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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

#include "Length.h"
#include "RenderStyleConstants.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderLayer;

// This class handles the auto-scrolling for <marquee>
class RenderMarquee final : public CanMakeCheckedPtr<RenderMarquee> {
    WTF_MAKE_TZONE_ALLOCATED(RenderMarquee);
    WTF_MAKE_NONCOPYABLE(RenderMarquee);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMarquee);
public:
    explicit RenderMarquee(RenderLayer*);
    ~RenderMarquee();

    bool isHorizontal() const;

    void start();
    void suspend();
    void stop();

    void updateMarqueeStyle();
    void updateMarqueePosition();

private:

    int speed() const { return m_speed; }
    int marqueeSpeed() const;

    MarqueeDirection direction() const;

    int computePosition(MarqueeDirection, bool stopAtClientEdge);

    void setEnd(int end) { m_end = end; }

    void timerFired();

    RenderLayer* m_layer;
    Timer m_timer;
    int m_currentLoop { 0 };
    int m_totalLoops { 0 };
    int m_start { 0 };
    int m_end { 0 };
    int m_speed { 0 };
    Length m_height;
    MarqueeDirection m_direction { MarqueeDirection::Auto };
    bool m_reset { false };
    bool m_suspended { false };
    bool m_stopped { false };
};

} // namespace WebCore
