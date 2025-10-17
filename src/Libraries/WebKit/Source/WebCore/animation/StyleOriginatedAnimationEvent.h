/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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

#include "AnimationEventBase.h"
#include "PseudoElementIdentifier.h"

namespace WebCore {

class StyleOriginatedAnimationEvent : public AnimationEventBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(StyleOriginatedAnimationEvent);
public:
    virtual ~StyleOriginatedAnimationEvent();

    double elapsedTime() const { return m_elapsedTime; }
    const String& pseudoElement();
    const std::optional<Style::PseudoElementIdentifier>& pseudoElementIdentifier() const { return m_pseudoElementIdentifier; }

protected:
    StyleOriginatedAnimationEvent(enum EventInterfaceType, const AtomString& type, WebAnimation*, std::optional<Seconds> scheduledTime, double, const std::optional<Style::PseudoElementIdentifier>&);
    StyleOriginatedAnimationEvent(enum EventInterfaceType, const AtomString&, const EventInit&, IsTrusted, double, const String&);

private:
    double m_elapsedTime;
    String m_pseudoElement;
    std::optional<Style::PseudoElementIdentifier> m_pseudoElementIdentifier { };
};

} // namespace WebCore
