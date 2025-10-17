/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
#include "StaticRange.h"

#include "ContainerNode.h"
#include "JSNode.h"
#include "Text.h"
#include "WebCoreOpaqueRootInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(StaticRange);

StaticRange::StaticRange(SimpleRange&& range)
    : SimpleRange(WTFMove(range))
{
}

Ref<StaticRange> StaticRange::create(SimpleRange&& range)
{
    return adoptRef(*new StaticRange(WTFMove(range)));
}

Ref<StaticRange> StaticRange::create(const SimpleRange& range)
{
    return create(SimpleRange { range });
}

static bool isDocumentTypeOrAttr(Node& node)
{
    // Before calling nodeType, do two fast non-virtual checks that cover almost all normal nodes, but are false for DocumentType and Attr.
    if (is<ContainerNode>(node) || is<Text>(node))
        return false;

    // Call nodeType explicitly and use a switch so we don't have to call it twice.
    switch (node.nodeType()) {
    case Node::ATTRIBUTE_NODE:
    case Node::DOCUMENT_TYPE_NODE:
        return true;
    default:
        return false;
    }
}

ExceptionOr<Ref<StaticRange>> StaticRange::create(Init&& init)
{
    ASSERT(init.startContainer);
    ASSERT(init.endContainer);
    if (isDocumentTypeOrAttr(*init.startContainer) || isDocumentTypeOrAttr(*init.endContainer))
        return Exception { ExceptionCode::InvalidNodeTypeError };
    return create({ { init.startContainer.releaseNonNull(), init.startOffset }, { init.endContainer.releaseNonNull(), init.endOffset } });
}

void StaticRange::visitNodesConcurrently(JSC::AbstractSlotVisitor& visitor) const
{
    addWebCoreOpaqueRoot(visitor, start.container.get());
    addWebCoreOpaqueRoot(visitor, end.container.get());
}

bool StaticRange::computeValidity() const
{
    Node& startContainer = this->startContainer();
    Node& endContainer = this->endContainer();

    if (!connectedInSameTreeScope(&startContainer.rootNode(), &endContainer.rootNode()))
        return false;
    if (startOffset() > startContainer.length())
        return false;
    if (endOffset() > endContainer.length())
        return false;
    if (&startContainer == &endContainer)
        return endOffset() > startOffset();
    return !is_gt(treeOrder<ComposedTree>(startContainer, endContainer));
}
} // namespace WebCore
