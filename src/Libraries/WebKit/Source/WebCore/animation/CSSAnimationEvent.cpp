/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include "CSSAnimationEvent.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSAnimationEvent);

CSSAnimationEvent::CSSAnimationEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
    : StyleOriginatedAnimationEvent(EventInterfaceType::CSSAnimationEvent, type, initializer, isTrusted, initializer.elapsedTime, initializer.pseudoElement)
    , m_animationName(initializer.animationName)
{
}

CSSAnimationEvent::CSSAnimationEvent(const AtomString& type, WebAnimation* animation, std::optional<Seconds> scheduledTime, double elapsedTime, const std::optional<Style::PseudoElementIdentifier>& pseudoElementIdentifier, const String& animationName)
    : StyleOriginatedAnimationEvent(EventInterfaceType::CSSAnimationEvent, type, animation, scheduledTime, elapsedTime, pseudoElementIdentifier)
    , m_animationName(animationName)
{
}

CSSAnimationEvent::~CSSAnimationEvent() = default;

} // namespace WebCore
