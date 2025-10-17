/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#include "StaticNodeList.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(StaticNodeList);
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(StaticWrapperNodeList);
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(StaticElementList);

unsigned StaticNodeList::length() const
{
    return m_nodes.size();
}

Node* StaticNodeList::item(unsigned index) const
{
    if (index < m_nodes.size())
        return const_cast<Node*>(m_nodes[index].ptr());
    return nullptr;
}

unsigned StaticWrapperNodeList::length() const
{
    return m_nodeList->length();
}

Node* StaticWrapperNodeList::item(unsigned index) const
{
    return m_nodeList->item(index);
}

unsigned StaticElementList::length() const
{
    return m_elements.size();
}

Element* StaticElementList::item(unsigned index) const
{
    if (index < m_elements.size())
        return const_cast<Element*>(m_elements[index].ptr());
    return nullptr;
}

} // namespace WebCore
