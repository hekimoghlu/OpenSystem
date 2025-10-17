/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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
#import "PlatformCALayerContentsDelayedReleaser.h"

#if PLATFORM(MAC)

#import "PlatformCALayer.h"
#import <wtf/RunLoop.h>

namespace WebCore {

PlatformCALayerContentsDelayedReleaser& PlatformCALayerContentsDelayedReleaser::singleton()
{
    static LazyNeverDestroyed<PlatformCALayerContentsDelayedReleaser> delayedReleaser;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        delayedReleaser.construct();
    });

    return delayedReleaser;
}

PlatformCALayerContentsDelayedReleaser::PlatformCALayerContentsDelayedReleaser() = default;

void PlatformCALayerContentsDelayedReleaser::takeLayerContents(PlatformCALayer& layer)
{
    ASSERT(isMainThread());

    auto retainedContents = RetainPtr { layer.contents() };
    if (retainedContents)
        m_retainedContents.append(WTFMove(retainedContents));
    layer.setContents(nullptr);
}

void PlatformCALayerContentsDelayedReleaser::mainThreadCommitWillStart()
{
    Locker locker { m_lock };
    ++m_inMainThreadCommitEntryCount;
    updateSawOverlappingCommit();
}

void PlatformCALayerContentsDelayedReleaser::mainThreadCommitDidEnd()
{
    bool bothCommitsDone;
    bool hadOverlappingCommit;
    {
        Locker locker { m_lock };
        ASSERT(m_inMainThreadCommitEntryCount);
        --m_inMainThreadCommitEntryCount;
        bothCommitsDone = !m_scrollingThreadCommitEntryCount && !m_inMainThreadCommitEntryCount;
        hadOverlappingCommit = m_hadOverlappingCommit;
    }

    if (bothCommitsDone) {
        if (hadOverlappingCommit && m_retainedContents.size()) {
            RunLoop::main().dispatch([] {
                PlatformCALayerContentsDelayedReleaser::singleton().clearRetainedContents();
            });
            return;
        }

        clearRetainedContents();
    }
}

void PlatformCALayerContentsDelayedReleaser::scrollingThreadCommitWillStart()
{
    ASSERT(!isMainThread());
    Locker locker { m_lock };
    ++m_scrollingThreadCommitEntryCount;
    updateSawOverlappingCommit();
}

void PlatformCALayerContentsDelayedReleaser::scrollingThreadCommitDidEnd()
{
    ASSERT(!isMainThread());
    Locker locker { m_lock };

    ASSERT(m_scrollingThreadCommitEntryCount);
    --m_scrollingThreadCommitEntryCount;
    if (!m_scrollingThreadCommitEntryCount && !m_inMainThreadCommitEntryCount) {
        if (m_hadOverlappingCommit) {
            // m_retainedContents might be empty (it's not protected by the lock so we can't check it here),
            // so this might be a pointless dispatch, but m_hadOverlappingCommit is rare.
            RunLoop::main().dispatch([] {
                PlatformCALayerContentsDelayedReleaser::singleton().clearRetainedContents();
            });
        }
    }
}

void PlatformCALayerContentsDelayedReleaser::updateSawOverlappingCommit()
{
    m_hadOverlappingCommit |= (m_inMainThreadCommitEntryCount && m_scrollingThreadCommitEntryCount);
}

void PlatformCALayerContentsDelayedReleaser::clearRetainedContents()
{
    ASSERT(isMainThread());
    m_retainedContents.clear();
}

} // namespace WebCore

#endif

