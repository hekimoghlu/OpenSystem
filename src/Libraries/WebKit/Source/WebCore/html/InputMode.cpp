/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
#include "InputMode.h"

#include "CommonAtomStrings.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

InputMode inputModeForAttributeValue(const AtomString& value)
{
    if (equalIgnoringASCIICase(value, InputModeNames::none()))
        return InputMode::None;
    if (equalIgnoringASCIICase(value, InputModeNames::text()))
        return InputMode::Text;
    if (equalIgnoringASCIICase(value, InputModeNames::tel()))
        return InputMode::Telephone;
    if (equalIgnoringASCIICase(value, InputModeNames::url()))
        return InputMode::Url;
    if (equalIgnoringASCIICase(value, InputModeNames::email()))
        return InputMode::Email;
    if (equalIgnoringASCIICase(value, InputModeNames::numeric()))
        return InputMode::Numeric;
    if (equalIgnoringASCIICase(value, InputModeNames::decimal()))
        return InputMode::Decimal;
    if (equalIgnoringASCIICase(value, InputModeNames::search()))
        return InputMode::Search;

    return InputMode::Unspecified;
}

const AtomString& stringForInputMode(InputMode mode)
{
    switch (mode) {
    case InputMode::Unspecified:
        return emptyAtom();
    case InputMode::None:
        return InputModeNames::none();
    case InputMode::Text:
        return InputModeNames::text();
    case InputMode::Telephone:
        return InputModeNames::tel();
    case InputMode::Url:
        return InputModeNames::url();
    case InputMode::Email:
        return InputModeNames::email();
    case InputMode::Numeric:
        return InputModeNames::numeric();
    case InputMode::Decimal:
        return InputModeNames::decimal();
    case InputMode::Search:
        return InputModeNames::search();
    }

    return emptyAtom();
}

namespace InputModeNames {

const AtomString& none()
{
    return noneAtom();
}

const AtomString& text()
{
    return textAtom();
}

const AtomString& tel()
{
    return telAtom();
}

const AtomString& url()
{
    return urlAtom();
}

const AtomString& email()
{
    return emailAtom();
}

const AtomString& numeric()
{
    static MainThreadNeverDestroyed<const AtomString> mode("numeric"_s);
    return mode;
}

const AtomString& decimal()
{
    static MainThreadNeverDestroyed<const AtomString> mode("decimal"_s);
    return mode;
}

const AtomString& search()
{
    return searchAtom();
}

} // namespace InputModeNames

} // namespace WebCore
