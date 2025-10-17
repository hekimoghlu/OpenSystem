/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#include "SetNodeAttributeCommand.h"

#include "CompositeEditCommand.h"
#include "Element.h"
#include <wtf/Assertions.h>

namespace WebCore {

SetNodeAttributeCommand::SetNodeAttributeCommand(Ref<Element>&& element, const QualifiedName& attribute, const AtomString& value)
    : SimpleEditCommand(element->document())
    , m_element(WTFMove(element))
    , m_attribute(attribute)
    , m_value(value)
{
}

void SetNodeAttributeCommand::doApply()
{
    auto element = protectedElement();
    m_oldValue = element->getAttribute(m_attribute);
    element->setAttribute(m_attribute, m_value);
}

void SetNodeAttributeCommand::doUnapply()
{
    protectedElement()->setAttribute(m_attribute, m_oldValue);
    m_oldValue = { };
}

#ifndef NDEBUG
void SetNodeAttributeCommand::getNodesInCommand(NodeSet& nodes)
{
    addNodeAndDescendants(protectedElement().ptr(), nodes);
}
#endif

} // namespace WebCore
