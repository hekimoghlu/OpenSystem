/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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
#include "SubmitInputType.h"

#include "DOMFormData.h"
#include "Document.h"
#include "ElementInlines.h"
#include "Event.h"
#include "HTMLFormElement.h"
#include "HTMLInputElement.h"
#include "InputTypeNames.h"
#include "LocalizedStrings.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SubmitInputType);

using namespace HTMLNames;

const AtomString& SubmitInputType::formControlType() const
{
    return InputTypeNames::submit();
}

bool SubmitInputType::appendFormData(DOMFormData& formData) const
{
    ASSERT(element());
    if (!element()->isActivatedSubmit())
        return false;
    formData.append(element()->name(), element()->valueWithDefault());
    if (auto& dirname = element()->attributeWithoutSynchronization(HTMLNames::dirnameAttr); !dirname.isNull())
        formData.append(dirname, element()->directionForFormData());
    return true;
}

bool SubmitInputType::supportsRequired() const
{
    return false;
}

void SubmitInputType::handleDOMActivateEvent(Event& event)
{
    ASSERT(element());
    Ref<HTMLInputElement> protectedElement(*element());
    if (protectedElement->isDisabledFormControl() || !protectedElement->form())
        return;

    Ref<HTMLFormElement> protectedForm(*protectedElement->form());

    // Update layout before processing form actions in case the style changes
    // the Form or button relationships.
    protectedElement->protectedDocument()->updateLayoutIgnorePendingStylesheets();

    if (RefPtr currentForm = protectedElement->form())
        currentForm->submitIfPossible(&event, element()); // Event handlers can run.
    event.setDefaultHandled();
}

bool SubmitInputType::canBeSuccessfulSubmitButton()
{
    return true;
}

String SubmitInputType::defaultValue() const
{
    return submitButtonDefaultLabel();
}

} // namespace WebCore
