/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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
#include "EmailInputType.h"

#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "HTMLParserIdioms.h"
#include "InputTypeNames.h"
#include "LocalizedStrings.h"
#include <JavaScriptCore/RegularExpression.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(EmailInputType);

using namespace HTMLNames;

// From https://html.spec.whatwg.org/#valid-e-mail-address.
static constexpr ASCIILiteral emailPattern = "^[a-zA-Z0-9.!#$%&'*+\\/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"_s;

static bool isValidEmailAddress(StringView address)
{
    int addressLength = address.length();
    if (!addressLength)
        return false;

    static NeverDestroyed<const JSC::Yarr::RegularExpression> regExp(StringView { emailPattern }, OptionSet<JSC::Yarr::Flags> { JSC::Yarr::Flags::IgnoreCase });

    int matchLength;
    int matchOffset = regExp.get().match(address, 0, &matchLength);

    return !matchOffset && matchLength == addressLength;
}

const AtomString& EmailInputType::formControlType() const
{
    return InputTypeNames::email();
}

bool EmailInputType::typeMismatchFor(const String& value) const
{
    ASSERT(element());
    if (value.isEmpty())
        return false;
    if (!element()->multiple())
        return !isValidEmailAddress(value);
    for (auto& address : value.splitAllowingEmptyEntries(',')) {
        if (!isValidEmailAddress(StringView(address).trim(isASCIIWhitespace<UChar>)))
            return true;
    }
    return false;
}

bool EmailInputType::typeMismatch() const
{
    ASSERT(element());
    return typeMismatchFor(element()->value());
}

String EmailInputType::typeMismatchText() const
{
    ASSERT(element());
    return element()->multiple() ? validationMessageTypeMismatchForMultipleEmailText() : validationMessageTypeMismatchForEmailText();
}

bool EmailInputType::supportsSelectionAPI() const
{
    return false;
}

void EmailInputType::attributeChanged(const QualifiedName& name)
{
    if (name == multipleAttr)
        element()->setValueInternal(sanitizeValue(element()->value()), TextFieldEventBehavior::DispatchNoEvent);

    BaseTextInputType::attributeChanged(name);
}

String EmailInputType::sanitizeValue(const String& proposedValue) const
{
    // Passing a lambda instead of a function name helps the compiler inline isHTMLLineBreak.
    String noLineBreakValue = proposedValue;
    if (UNLIKELY(containsHTMLLineBreak(proposedValue))) {
        noLineBreakValue = proposedValue.removeCharacters([](auto character) {
            return isHTMLLineBreak(character);
        });
    }

    ASSERT(element());
    if (!element()->multiple())
        return noLineBreakValue.trim(isASCIIWhitespace);
    Vector<String> addresses = noLineBreakValue.splitAllowingEmptyEntries(',');
    StringBuilder strippedValue;
    for (unsigned i = 0; i < addresses.size(); ++i) {
        if (i > 0)
            strippedValue.append(',');
        strippedValue.append(addresses[i].trim(isASCIIWhitespace));
    }
    return strippedValue.toString();
}

} // namespace WebCore
