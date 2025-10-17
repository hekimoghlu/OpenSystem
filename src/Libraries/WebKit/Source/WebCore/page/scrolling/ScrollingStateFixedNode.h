/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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

#include "ScrollingConstraints.h"
#include "ScrollingStateNode.h"

#include <wtf/Forward.h>

namespace WebCore {

class FixedPositionViewportConstraints;

class ScrollingStateFixedNode final : public ScrollingStateNode {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingStateFixedNode, WEBCORE_EXPORT);
public:
    template<typename... Args> static Ref<ScrollingStateFixedNode> create(Args&&... args) { return adoptRef(*new ScrollingStateFixedNode(std::forward<Args>(args)...)); }

    Ref<ScrollingStateNode> clone(ScrollingStateTree&) final;

    virtual ~ScrollingStateFixedNode();

    WEBCORE_EXPORT void updateConstraints(const FixedPositionViewportConstraints&);
    const FixedPositionViewportConstraints& viewportConstraints() const { return m_constraints; }

private:
    WEBCORE_EXPORT ScrollingStateFixedNode(ScrollingNodeID, Vector<Ref<ScrollingStateNode>>&&, OptionSet<ScrollingStateNodeProperty>, std::optional<PlatformLayerIdentifier>, FixedPositionViewportConstraints&&);
    ScrollingStateFixedNode(ScrollingStateTree&, ScrollingNodeID);
    ScrollingStateFixedNode(const ScrollingStateFixedNode&, ScrollingStateTree&);

    void reconcileLayerPositionForViewportRect(const LayoutRect& viewportRect, ScrollingLayerPositionAction) final;

    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const final;
    OptionSet<ScrollingStateNode::Property> applicableProperties() const final;

    FixedPositionViewportConstraints m_constraints;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_STATE_NODE(ScrollingStateFixedNode, isFixedNode())

#endif // ENABLE(ASYNC_SCROLLING)
