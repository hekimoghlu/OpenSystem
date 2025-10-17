/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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

#if ENABLE(UI_SIDE_COMPOSITING)

#include <WebCore/FrameIdentifier.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class ScrollingStateTree;
}

namespace WebKit {

class RemoteScrollingCoordinatorTransaction {
public:
    enum class FromDeserialization : bool { No, Yes };
    RemoteScrollingCoordinatorTransaction();
    RemoteScrollingCoordinatorTransaction(std::unique_ptr<WebCore::ScrollingStateTree>&&, bool, std::optional<WebCore::FrameIdentifier> = std::nullopt, FromDeserialization = FromDeserialization::Yes);
    RemoteScrollingCoordinatorTransaction(RemoteScrollingCoordinatorTransaction&&);
    RemoteScrollingCoordinatorTransaction& operator=(RemoteScrollingCoordinatorTransaction&&);
    ~RemoteScrollingCoordinatorTransaction();

    std::unique_ptr<WebCore::ScrollingStateTree>& scrollingStateTree() { return m_scrollingStateTree; }
    const std::unique_ptr<WebCore::ScrollingStateTree>& scrollingStateTree() const { return m_scrollingStateTree; }

    std::optional<WebCore::FrameIdentifier> rootFrameIdentifier() const { return m_rootFrameID; }
    void setFrameIdentifier(WebCore::FrameIdentifier identifier) { m_rootFrameID = identifier; }

    bool clearScrollLatching() const { return m_clearScrollLatching; }

#if !defined(NDEBUG) || !LOG_DISABLED
    String description() const;
    void dump() const;
#endif

private:
    std::unique_ptr<WebCore::ScrollingStateTree> m_scrollingStateTree;
    
    // Data encoded here should be "imperative" (valid just for one transaction). Stateful things should live on scrolling tree nodes.
    // Maybe RequestedScrollData should move here.
    bool m_clearScrollLatching { false };

    // Frame Identifier for the root frame of this transaction
    Markable<WebCore::FrameIdentifier> m_rootFrameID;
};

} // namespace WebKit

#endif // ENABLE(UI_SIDE_COMPOSITING)
