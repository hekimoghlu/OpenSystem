/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#import "GestureRecognizerConsistencyEnforcer.h"

#if PLATFORM(IOS_FAMILY)

#import "Logging.h"
#import "WKContentViewInteraction.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(GestureRecognizerConsistencyEnforcer);

GestureRecognizerConsistencyEnforcer::GestureRecognizerConsistencyEnforcer(WKContentView *view)
    : m_view(view)
    , m_timer(RunLoop::main(), this, &GestureRecognizerConsistencyEnforcer::timerFired)
{
    ASSERT(m_view);
}

GestureRecognizerConsistencyEnforcer::~GestureRecognizerConsistencyEnforcer() = default;

void GestureRecognizerConsistencyEnforcer::beginTracking(WKDeferringGestureRecognizer *gesture)
{
    m_timer.stop();
    m_deferringGestureRecognizersWithTouches.add(gesture);
}

void GestureRecognizerConsistencyEnforcer::endTracking(WKDeferringGestureRecognizer *gesture)
{
    if (!m_deferringGestureRecognizersWithTouches.remove(gesture))
        return;

    if (m_deferringGestureRecognizersWithTouches.isEmpty())
        m_timer.startOneShot(1_s);
}

void GestureRecognizerConsistencyEnforcer::reset()
{
    m_timer.stop();
    m_deferringGestureRecognizersWithTouches.clear();
}

void GestureRecognizerConsistencyEnforcer::timerFired()
{
    auto strongView = m_view.get();
    auto possibleDeferringGestures = [NSMutableArray<WKDeferringGestureRecognizer *> array];
    for (WKDeferringGestureRecognizer *gesture in [strongView deferringGestures]) {
        if (gesture.state == UIGestureRecognizerStatePossible && gesture.enabled)
            [possibleDeferringGestures addObject:gesture];
    }

    if (!possibleDeferringGestures.count)
        return;

    auto touchEventState = [strongView touchEventGestureRecognizer].state;
    if (touchEventState == UIGestureRecognizerStatePossible || touchEventState == UIGestureRecognizerStateBegan || touchEventState == UIGestureRecognizerStateChanged)
        return;

    for (WKDeferringGestureRecognizer *gesture in possibleDeferringGestures)
        [gesture setState:UIGestureRecognizerStateEnded];

    RELEASE_LOG_FAULT(ViewGestures, "Touch event gesture recognizer failed to reset after ending gesture deferral: %@", possibleDeferringGestures);
}

void GestureRecognizerConsistencyEnforcer::ref() const
{
    auto strongView = m_view.get();
    [strongView retain];
}

void GestureRecognizerConsistencyEnforcer::deref() const
{
    auto strongView = m_view.get();
    [strongView release];
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
