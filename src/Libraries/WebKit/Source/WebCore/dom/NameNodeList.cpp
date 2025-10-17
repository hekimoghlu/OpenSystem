/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
#include "NameNodeList.h"

#include "ElementInlines.h"
#include "LiveNodeListInlines.h"
#include "NodeRareData.h"
#include "Quirks.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace HTMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(NameNodeList);

NameNodeList::NameNodeList(ContainerNode& rootNode, const AtomString& name)
    : CachedLiveNodeList(rootNode, NodeListInvalidationType::InvalidateOnNameAttrChange)
    , m_name(name)
    , m_needsGetElementsByNameQuirk(rootNode.document().quirks().needsGetElementsByNameQuirk())
{
}

Ref<NameNodeList> NameNodeList::create(ContainerNode& rootNode, const AtomString& name)
{
    return adoptRef(*new NameNodeList(rootNode, name));
}

NameNodeList::~NameNodeList()
{
    protectedOwnerNode()->nodeLists()->removeCacheWithAtomName(*this, m_name);
}

bool NameNodeList::elementMatches(Element& element) const
{
    return (is<HTMLElement>(element) || m_needsGetElementsByNameQuirk) && element.getNameAttribute() == m_name;
}

} // namespace WebCore
