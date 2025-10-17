/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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

#if PLATFORM(IOS_FAMILY)

#import <wtf/FastMalloc.h>
#import <wtf/Noncopyable.h>
#import <wtf/RetainPtr.h>
#import <wtf/RunLoop.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>

@class WKContentView;
@class WKDeferringGestureRecognizer;

namespace WebKit {

class GestureRecognizerConsistencyEnforcer {
    WTF_MAKE_TZONE_ALLOCATED(GestureRecognizerConsistencyEnforcer);
    WTF_MAKE_NONCOPYABLE(GestureRecognizerConsistencyEnforcer);
public:
    GestureRecognizerConsistencyEnforcer(WKContentView *);
    ~GestureRecognizerConsistencyEnforcer();

    void ref() const;
    void deref() const;

    void beginTracking(WKDeferringGestureRecognizer *);
    void endTracking(WKDeferringGestureRecognizer *);

    void reset();

private:
    void timerFired();

    WeakObjCPtr<WKContentView> m_view; // Cannot be null.
    RunLoop::Timer m_timer;
    HashSet<RetainPtr<WKDeferringGestureRecognizer>> m_deferringGestureRecognizersWithTouches;
};

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
