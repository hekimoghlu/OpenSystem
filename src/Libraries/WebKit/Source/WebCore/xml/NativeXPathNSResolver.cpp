/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
#include "NativeXPathNSResolver.h"

#include "CommonAtomStrings.h"
#include "Node.h"
#include "XMLNames.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

NativeXPathNSResolver::NativeXPathNSResolver(Ref<Node>&& node)
    : m_node(WTFMove(node))
{
}

NativeXPathNSResolver::~NativeXPathNSResolver() = default;

AtomString NativeXPathNSResolver::lookupNamespaceURI(const AtomString& prefix)
{
    // This is not done by Node::lookupNamespaceURI as per the DOM3 Core spec,
    // but the XPath spec says that we should do it for XPathNSResolver.
    if (prefix == xmlAtom())
        return XMLNames::xmlNamespaceURI.get();
    
    return m_node->lookupNamespaceURI(prefix);
}

} // namespace WebCore
