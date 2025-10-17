/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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
#include "Traversal.h"

#include "CallbackResult.h"
#include "Node.h"
#include "NodeFilter.h"
#include <wtf/SetForScope.h>

namespace WebCore {

NodeIteratorBase::NodeIteratorBase(Node& rootNode, unsigned whatToShow, RefPtr<NodeFilter>&& nodeFilter)
    : m_root(rootNode)
    , m_filter(WTFMove(nodeFilter))
    , m_whatToShow(whatToShow)
{
}

// https://dom.spec.whatwg.org/#concept-node-filter
ExceptionOr<unsigned short> NodeIteratorBase::acceptNodeSlowCase(Node& node)
{
    ASSERT(m_filter);
    if (m_isActive)
        return Exception { ExceptionCode::InvalidStateError, "Recursive filters are not allowed"_s };

    if (!matchesWhatToShow(node))
        return NodeFilter::FILTER_SKIP;

    SetForScope isActive(m_isActive, true);
    auto callbackResult = m_filter->acceptNodeRethrowingException(node);
    if (callbackResult.type() == CallbackResultType::ExceptionThrown)
        return Exception { ExceptionCode::ExistingExceptionError };
    return callbackResult.releaseReturnValue();
}

} // namespace WebCore
