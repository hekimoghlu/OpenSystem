/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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
#include "StyleOriginatedAnimationEvent.h"

#include "Node.h"
#include "WebAnimationUtilities.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(StyleOriginatedAnimationEvent);

StyleOriginatedAnimationEvent::StyleOriginatedAnimationEvent(enum EventInterfaceType eventInterface, const AtomString& type, WebAnimation* animation, std::optional<Seconds> scheduledTime, double elapsedTime, const std::optional<Style::PseudoElementIdentifier>& pseudoElementIdentifier)
    : AnimationEventBase(eventInterface, type, animation, scheduledTime)
    , m_elapsedTime(elapsedTime)
    , m_pseudoElementIdentifier(pseudoElementIdentifier)
{
}

StyleOriginatedAnimationEvent::StyleOriginatedAnimationEvent(enum EventInterfaceType eventInterface, const AtomString& type, const EventInit& init, IsTrusted isTrusted, double elapsedTime, const String& pseudoElement)
    : AnimationEventBase(eventInterface, type, init, isTrusted)
    , m_elapsedTime(elapsedTime)
    , m_pseudoElement(pseudoElement)
{
    auto* node = dynamicDowncast<Node>(target());
    auto [parsed, pseudoElementIdentifier] = pseudoElementIdentifierFromString(m_pseudoElement, node ? &node->document() : nullptr);
    m_pseudoElementIdentifier = parsed ? pseudoElementIdentifier : std::nullopt;
}

StyleOriginatedAnimationEvent::~StyleOriginatedAnimationEvent() = default;

const String& StyleOriginatedAnimationEvent::pseudoElement()
{
    if (m_pseudoElementIdentifier && m_pseudoElement.isNull())
        m_pseudoElement = pseudoElementIdentifierAsString(m_pseudoElementIdentifier);
    return m_pseudoElement;
}

} // namespace WebCore
