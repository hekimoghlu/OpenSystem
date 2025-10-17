/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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

#include "ExceptionOr.h"
#include "Node.h"
#include "NodeFilter.h"
#include <wtf/RefPtr.h>

namespace WebCore {

class NodeIteratorBase {
public:
    Node& root() { return m_root.get(); }
    const Node& root() const { return m_root.get(); }
    Ref<Node> protectedRoot() const { return m_root; }

    unsigned whatToShow() const { return m_whatToShow; }
    NodeFilter* filter() const { return m_filter.get(); }

protected:
    NodeIteratorBase(Node&, unsigned whatToShow, RefPtr<NodeFilter>&&);
    ExceptionOr<unsigned short> acceptNode(Node& node)
    {
        // https://dom.spec.whatwg.org/#concept-node-filter
        if (!m_filter)
            return matchesWhatToShow(node) ? NodeFilter::FILTER_ACCEPT : NodeFilter::FILTER_SKIP;
        return acceptNodeSlowCase(node);
    }

    bool matchesWhatToShow(const Node& node) const
    {
        unsigned nodeMask = 1 << (node.nodeType() - 1);
        return nodeMask & m_whatToShow;
    }

private:
    ExceptionOr<unsigned short> acceptNodeSlowCase(Node&);

    Ref<Node> m_root;
    RefPtr<NodeFilter> m_filter;
    unsigned m_whatToShow;
    bool m_isActive { false };
};

} // namespace WebCore
