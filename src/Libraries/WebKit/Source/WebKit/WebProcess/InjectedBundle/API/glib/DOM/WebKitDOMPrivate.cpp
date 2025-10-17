/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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
#include "WebKitDOMPrivate.h"

#include "WebKitDOMDocumentPrivate.h"
#include "WebKitDOMElementPrivate.h"
#include "WebKitDOMNodePrivate.h"
#include <WebCore/HTMLFormElement.h>

#if PLATFORM(GTK)
#include "WebKitDOMPrivateGtk.h"
#endif

namespace WebKit {

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

WebKitDOMNode* wrap(WebCore::Node* node)
{
    ASSERT(node);
    ASSERT(node->nodeType());

#if PLATFORM(GTK)
    if (auto* wrapper = wrapNodeGtk(node))
        return wrapper;
#endif

    switch (node->nodeType()) {
    case WebCore::Node::ELEMENT_NODE:
        return WEBKIT_DOM_NODE(wrapElement(downcast<WebCore::Element>(node)));
    case WebCore::Node::DOCUMENT_NODE:
        return WEBKIT_DOM_NODE(wrapDocument(downcast<WebCore::Document>(node)));
    case WebCore::Node::ATTRIBUTE_NODE:
    case WebCore::Node::TEXT_NODE:
    case WebCore::Node::CDATA_SECTION_NODE:
    case WebCore::Node::PROCESSING_INSTRUCTION_NODE:
    case WebCore::Node::COMMENT_NODE:
    case WebCore::Node::DOCUMENT_TYPE_NODE:
    case WebCore::Node::DOCUMENT_FRAGMENT_NODE:
        break;
    }

    return wrapNode(node);
}

G_GNUC_END_IGNORE_DEPRECATIONS;

} // namespace WebKit
