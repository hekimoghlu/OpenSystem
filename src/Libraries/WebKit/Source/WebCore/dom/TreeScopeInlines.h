/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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

#include "ContainerNode.h"
#include "TreeScopeOrderedMap.h"

namespace WebCore {

inline IdTargetObserverRegistry& TreeScope::idTargetObserverRegistry()
{
    if (m_idTargetObserverRegistry)
        return *m_idTargetObserverRegistry;
    return ensureIdTargetObserverRegistry();
}

inline Ref<ContainerNode> TreeScope::protectedRootNode() const
{
    return rootNode();
}

inline bool TreeScope::hasElementWithId(const AtomString& id) const
{
    return m_elementsById && m_elementsById->contains(id);
}

inline bool TreeScope::containsMultipleElementsWithId(const AtomString& id) const
{
    return m_elementsById && !id.isEmpty() && m_elementsById->containsMultiple(id);
}

inline bool TreeScope::hasElementWithName(const AtomString& id) const
{
    return m_elementsByName && m_elementsByName->contains(id);
}

inline bool TreeScope::containsMultipleElementsWithName(const AtomString& name) const
{
    return m_elementsByName && !name.isEmpty() && m_elementsByName->containsMultiple(name);
}

} // namespace WebCore
