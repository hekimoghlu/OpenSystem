/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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

class ScrollingStatePluginHostingNode final : public ScrollingStateNode {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingStatePluginHostingNode, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT static Ref<ScrollingStatePluginHostingNode> create(ScrollingStateTree&, ScrollingNodeID);
    WEBCORE_EXPORT static Ref<ScrollingStatePluginHostingNode> create(ScrollingNodeID, Vector<Ref<ScrollingStateNode>>&&, OptionSet<ScrollingStateNodeProperty>, std::optional<PlatformLayerIdentifier>);
    Ref<ScrollingStateNode> clone(ScrollingStateTree&) override;

    virtual ~ScrollingStatePluginHostingNode();

    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const override;

private:
    ScrollingStatePluginHostingNode(ScrollingNodeID, Vector<Ref<ScrollingStateNode>>&&, OptionSet<ScrollingStateNodeProperty>, std::optional<PlatformLayerIdentifier>);
    ScrollingStatePluginHostingNode(ScrollingStateTree&, ScrollingNodeID);
    ScrollingStatePluginHostingNode(const ScrollingStatePluginHostingNode&, ScrollingStateTree&);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_STATE_NODE(ScrollingStatePluginHostingNode, isPluginHostingNode())

#endif // ENABLE(ASYNC_SCROLLING)
