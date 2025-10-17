/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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
#include "AutoFillButtonElement.h"

#include "Event.h"
#include "EventNames.h"
#include "HTMLNames.h"
#include "MouseEvent.h"
#include "TextFieldInputType.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AutoFillButtonElement);

using namespace HTMLNames;

Ref<AutoFillButtonElement> AutoFillButtonElement::create(Document& document, AutoFillButtonOwner& owner)
{
    return adoptRef(*new AutoFillButtonElement(document, owner));
}

AutoFillButtonElement::AutoFillButtonElement(Document& document, AutoFillButtonOwner& owner)
    : HTMLDivElement(divTag, document)
    , m_owner(owner)
{
}

void AutoFillButtonElement::defaultEventHandler(Event& event)
{
    auto* mouseEvent = dynamicDowncast<MouseEvent>(event);
    if (!mouseEvent) {
        if (!event.defaultHandled())
            HTMLDivElement::defaultEventHandler(event);
        return;
    }

    if (isAnyClick(*mouseEvent)) {
        m_owner.autoFillButtonElementWasClicked();
        event.setDefaultHandled();
    }

    if (!event.defaultHandled())
        HTMLDivElement::defaultEventHandler(event);
}

} // namespace WebCore
