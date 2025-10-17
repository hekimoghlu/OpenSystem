/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
#include "ViewSnapshotStore.h"

#include "WebBackForwardList.h"
#include "WebPageProxy.h"
#include <wtf/NeverDestroyed.h>

#if PLATFORM(IOS_FAMILY)
static const size_t maximumSnapshotCacheSize = 50 * (1024 * 1024);
#else
static const size_t maximumSnapshotCacheSize = 400 * (1024 * 1024);
#endif

namespace WebKit {
using namespace WebCore;

ViewSnapshotStore::ViewSnapshotStore()
{
}

ViewSnapshotStore::~ViewSnapshotStore()
{
    discardSnapshotImages();
}

ViewSnapshotStore& ViewSnapshotStore::singleton()
{
    static NeverDestroyed<ViewSnapshotStore> store;
    return store;
}

void ViewSnapshotStore::didAddImageToSnapshot(ViewSnapshot& snapshot)
{
    bool isNewEntry = m_snapshotsWithImages.add(snapshot).isNewEntry;
    ASSERT_UNUSED(isNewEntry, isNewEntry);
    m_snapshotCacheSize += snapshot.estimatedImageSizeInBytes();
}

void ViewSnapshotStore::willRemoveImageFromSnapshot(ViewSnapshot& snapshot)
{
    bool removed = m_snapshotsWithImages.remove(snapshot);
    ASSERT_UNUSED(removed, removed);
    m_snapshotCacheSize -= snapshot.estimatedImageSizeInBytes();
}

void ViewSnapshotStore::pruneSnapshots(WebPageProxy& webPageProxy)
{
    if (m_snapshotCacheSize <= maximumSnapshotCacheSize)
        return;

    ASSERT(!m_snapshotsWithImages.isEmpty());

    // FIXME: We have enough information to do smarter-than-LRU eviction (making use of the back-forward lists, etc.)

    m_snapshotsWithImages.first()->clearImage();
}

void ViewSnapshotStore::recordSnapshot(WebPageProxy& webPageProxy, WebBackForwardListItem& item)
{
    if (webPageProxy.isShowingNavigationGestureSnapshot())
        return;

    pruneSnapshots(webPageProxy);

    webPageProxy.willRecordNavigationSnapshot(item);

    auto snapshot = webPageProxy.takeViewSnapshot(std::nullopt);
    if (!snapshot)
        return;

#if PLATFORM(MAC)
    snapshot->setVolatile(true);
#endif
    snapshot->setRenderTreeSize(webPageProxy.renderTreeSize());
    snapshot->setDeviceScaleFactor(webPageProxy.deviceScaleFactor());
    snapshot->setBackgroundColor(webPageProxy.pageExtendedBackgroundColor());
    snapshot->setViewScrollPosition(WebCore::roundedIntPoint(webPageProxy.viewScrollPosition()));
    snapshot->setOrigin(WebCore::SecurityOriginData::fromURL(URL(item.url())));

    item.setSnapshot(WTFMove(snapshot));
}

void ViewSnapshotStore::discardSnapshotImages()
{
    while (!m_snapshotsWithImages.isEmpty())
        m_snapshotsWithImages.first()->clearImage();
}

void ViewSnapshotStore::discardSnapshotImagesForOrigin(const WebCore::SecurityOriginData& origin)
{
    for (auto it = m_snapshotsWithImages.begin(); it != m_snapshotsWithImages.end();) {
        auto viewSnapshot = *it;
        ++it;
        if (viewSnapshot->origin() == origin)
            viewSnapshot->clearImage();
    }
}

ViewSnapshot::~ViewSnapshot()
{
    clearImage();
}

} // namespace WebKit
