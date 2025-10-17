/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
#include "InputMethodState.h"

#include "ArgumentCoders.h"
#include <WebCore/ElementInlines.h>
#include <WebCore/HTMLInputElement.h>

namespace WebKit {

void InputMethodState::setPurposeOrHintForInputMode(WebCore::InputMode inputMode)
{
    switch (inputMode) {
    case WebCore::InputMode::None:
        hints.add(InputMethodState::Hint::InhibitOnScreenKeyboard);
        break;
    case WebCore::InputMode::Unspecified:
    case WebCore::InputMode::Text:
        purpose = Purpose::FreeForm;
        break;
    case WebCore::InputMode::Telephone:
        purpose = Purpose::Phone;
        break;
    case WebCore::InputMode::Url:
        purpose = Purpose::Url;
        break;
    case WebCore::InputMode::Email:
        purpose = Purpose::Email;
        break;
    case WebCore::InputMode::Numeric:
        purpose = Purpose::Digits;
        break;
    case WebCore::InputMode::Decimal:
        purpose = Purpose::Number;
        break;
    case WebCore::InputMode::Search:
        break;
    }
}

static bool inputElementHasDigitsPattern(WebCore::HTMLInputElement& element)
{
    const auto& pattern = element.attributeWithoutSynchronization(WebCore::HTMLNames::patternAttr);
    return pattern == "\\d*"_s || pattern == "[0-9]*"_s;
}

void InputMethodState::setPurposeForInputElement(WebCore::HTMLInputElement& element)
{
    if (element.isPasswordField())
        purpose = Purpose::Password;
    else if (element.isEmailField())
        purpose = Purpose::Email;
    else if (element.isTelephoneField())
        purpose = Purpose::Phone;
    else if (element.isNumberField())
        purpose = inputElementHasDigitsPattern(element) ? Purpose::Digits : Purpose::Number;
    else if (element.isURLField())
        purpose = Purpose::Url;
    else if (element.isText() && inputElementHasDigitsPattern(element))
        purpose = Purpose::Digits;
}

void InputMethodState::addHintsForAutocapitalizeType(WebCore::AutocapitalizeType autocapitalizeType)
{
    switch (autocapitalizeType) {
    case WebCore::AutocapitalizeType::Default:
        break;
    case WebCore::AutocapitalizeType::None:
        hints.add(InputMethodState::Hint::Lowercase);
        break;
    case WebCore::AutocapitalizeType::Words:
        hints.add(InputMethodState::Hint::UppercaseWords);
        break;
    case WebCore::AutocapitalizeType::Sentences:
        hints.add(InputMethodState::Hint::UppercaseSentences);
        break;
    case WebCore::AutocapitalizeType::AllCharacters:
        hints.add(InputMethodState::Hint::UppercaseChars);
        break;
    }
}

} // namespace WebKit
