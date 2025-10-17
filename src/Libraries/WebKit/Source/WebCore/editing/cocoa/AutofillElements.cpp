/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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
#include "AutofillElements.h"

#include "FocusController.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AutofillElements);

static inline bool isAutofillableElement(Element& node)
{
    auto* inputElement = dynamicDowncast<HTMLInputElement>(node);
    return inputElement && (inputElement->isTextField() || inputElement->isEmailField());
}

static inline RefPtr<HTMLInputElement> nextAutofillableElement(Node* startNode, FocusController& focusController)
{
    RefPtr nextElement = dynamicDowncast<Element>(startNode);
    if (!nextElement)
        return nullptr;

    do {
        nextElement = focusController.nextFocusableElement(*nextElement.get());
    } while (nextElement && !isAutofillableElement(*nextElement.get()));

    if (!nextElement)
        return nullptr;

    return &downcast<HTMLInputElement>(*nextElement);
}

static inline RefPtr<HTMLInputElement> previousAutofillableElement(Node* startNode, FocusController& focusController)
{
    RefPtr previousElement = dynamicDowncast<Element>(startNode);
    if (!previousElement)
        return nullptr;

    do {
        previousElement = focusController.previousFocusableElement(*previousElement.get());
    } while (previousElement && !isAutofillableElement(*previousElement.get()));

    if (!previousElement)
        return nullptr;
    
    return &downcast<HTMLInputElement>(*previousElement);
}

AutofillElements::AutofillElements(RefPtr<HTMLInputElement>&& username, RefPtr<HTMLInputElement>&& password, RefPtr<HTMLInputElement>&& secondPassword)
    : m_username(WTFMove(username))
    , m_password(WTFMove(password))
    , m_secondPassword(WTFMove(secondPassword))
{
}

std::optional<AutofillElements> AutofillElements::computeAutofillElements(Ref<HTMLInputElement> start)
{
    if (!start->document().page())
        return std::nullopt;
    CheckedRef focusController = { start->document().page()->focusController() };
    if (start->isPasswordField()) {
        auto previousElement = previousAutofillableElement(start.ptr(), focusController);
        auto nextElement = nextAutofillableElement(start.ptr(), focusController);

        bool previousFieldIsTextField = previousElement && !previousElement->isPasswordField();
        bool hasSecondPasswordFieldToFill = nextElement && nextElement->isPasswordField() && nextElement->value().isEmpty();

        // Always allow AutoFill in a password field, even if we fill information only into it.
        return {{ previousFieldIsTextField ? WTFMove(previousElement) : nullptr, WTFMove(start), hasSecondPasswordFieldToFill ? WTFMove(nextElement) : nullptr }};
    } else {
        RefPtr<HTMLInputElement> nextElement = nextAutofillableElement(start.ptr(), focusController);
        if (nextElement && is<HTMLInputElement>(*nextElement)) {
            if (nextElement->isPasswordField()) {
                auto elementAfterNextElement = nextAutofillableElement(nextElement.get(), focusController);
                bool hasSecondPasswordFieldToFill = elementAfterNextElement && elementAfterNextElement->isPasswordField() && elementAfterNextElement->value().isEmpty();

                return {{ WTFMove(start), WTFMove(nextElement), hasSecondPasswordFieldToFill ? WTFMove(elementAfterNextElement) : nullptr }};
            }
        }
    }

    // Handle the case where a username field appears separately from a password field.
    auto autofillData = start->autofillData();
    if (toAutofillFieldName(autofillData.fieldName) == AutofillFieldName::Username || toAutofillFieldName(autofillData.fieldName) == AutofillFieldName::WebAuthn)
        return {{ WTFMove(start), nullptr, nullptr }};

    return std::nullopt;
}

void AutofillElements::autofill(String username, String password)
{
    if (m_username)
        m_username->setValueForUser(username);
    if (m_password)
        m_password->setValueForUser(password);
    if (m_secondPassword)
        m_secondPassword->setValueForUser(password);
}

} // namespace WebCore
