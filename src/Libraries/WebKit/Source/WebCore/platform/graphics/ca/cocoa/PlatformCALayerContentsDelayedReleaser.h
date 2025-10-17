/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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

#if PLATFORM(MAC)

#include <wtf/Lock.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class PlatformCALayer;

// This class exists to work around rdar://85892959, where CABackingStore objects would get released on the ScrollingThread
// during scrolling commits, which can take long enough to cause scrolling frame drops.
class PlatformCALayerContentsDelayedReleaser : ThreadSafeRefCounted<PlatformCALayerContentsDelayedReleaser> {
    WTF_MAKE_NONCOPYABLE(PlatformCALayerContentsDelayedReleaser);
public:
    static PlatformCALayerContentsDelayedReleaser& singleton();

    void takeLayerContents(PlatformCALayer&);
    
    void mainThreadCommitWillStart();
    void mainThreadCommitDidEnd();

    void scrollingThreadCommitWillStart();
    void scrollingThreadCommitDidEnd();

private:
    friend LazyNeverDestroyed<PlatformCALayerContentsDelayedReleaser>;
    
    PlatformCALayerContentsDelayedReleaser();

    void updateSawOverlappingCommit() WTF_REQUIRES_LOCK(m_lock);
    void clearRetainedContents();

    Vector<RetainPtr<CFTypeRef>> m_retainedContents;

    Lock m_lock;
    unsigned m_inMainThreadCommitEntryCount WTF_GUARDED_BY_LOCK(m_lock) { 0 };
    unsigned m_scrollingThreadCommitEntryCount WTF_GUARDED_BY_LOCK(m_lock) { 0 };
    bool m_hadOverlappingCommit WTF_GUARDED_BY_LOCK(m_lock) { false };
};

} // namespace WebCore

#endif // PLATFORM(MAC)
