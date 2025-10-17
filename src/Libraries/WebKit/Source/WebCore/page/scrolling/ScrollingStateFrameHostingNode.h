/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingStateNode.h"

namespace WebCore {

class Scrollbar;

class ScrollingStateFrameHostingNode final : public ScrollingStateNode {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingStateFrameHostingNode, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT static Ref<ScrollingStateFrameHostingNode> create(ScrollingStateTree&, ScrollingNodeID);
    WEBCORE_EXPORT static Ref<ScrollingStateFrameHostingNode> create(ScrollingNodeID, Vector<Ref<ScrollingStateNode>>&&, OptionSet<ScrollingStateNodeProperty>, std::optional<PlatformLayerIdentifier>, std::optional<LayerHostingContextIdentifier>);
    Ref<ScrollingStateNode> clone(ScrollingStateTree&) override;

    std::optional<LayerHostingContextIdentifier> layerHostingContextIdentifier() const { return m_hostingContext; }
    void setLayerHostingContextIdentifier(const std::optional<LayerHostingContextIdentifier>);

    virtual ~ScrollingStateFrameHostingNode();

    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const override;

private:
    ScrollingStateFrameHostingNode(ScrollingNodeID, Vector<Ref<ScrollingStateNode>>&&, OptionSet<ScrollingStateNodeProperty>, std::optional<PlatformLayerIdentifier>, std::optional<LayerHostingContextIdentifier>);
    ScrollingStateFrameHostingNode(ScrollingStateTree&, ScrollingNodeID);
    ScrollingStateFrameHostingNode(const ScrollingStateFrameHostingNode&, ScrollingStateTree&);

    std::optional<LayerHostingContextIdentifier> m_hostingContext;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_STATE_NODE(ScrollingStateFrameHostingNode, isFrameHostingNode())

#endif // ENABLE(ASYNC_SCROLLING)
