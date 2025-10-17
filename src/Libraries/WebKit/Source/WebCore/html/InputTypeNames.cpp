/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#include "InputTypeNames.h"

#include "CommonAtomStrings.h"
#include "HTMLNames.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

namespace InputTypeNames {

// The type names must be lowercased because they will be the return values of
// input.type and input.type must be lowercase according to DOM Level 2.

const AtomString& button()
{
    return HTMLNames::buttonTag->localName();
}

const AtomString& checkbox()
{
    static MainThreadNeverDestroyed<const AtomString> name("checkbox"_s);
    return name;
}

const AtomString& color()
{
    static MainThreadNeverDestroyed<const AtomString> name("color"_s);
    return name;
}

const AtomString& date()
{
    static MainThreadNeverDestroyed<const AtomString> name("date"_s);
    return name;
}

const AtomString& datetimelocal()
{
    static MainThreadNeverDestroyed<const AtomString> name("datetime-local"_s);
    return name;
}

const AtomString& email()
{
    return emailAtom();
}

const AtomString& file()
{
    static MainThreadNeverDestroyed<const AtomString> name("file"_s);
    return name;
}

const AtomString& hidden()
{
    static MainThreadNeverDestroyed<const AtomString> name("hidden"_s);
    return name;
}

const AtomString& image()
{
    static MainThreadNeverDestroyed<const AtomString> name("image"_s);
    return name;
}

const AtomString& month()
{
    static MainThreadNeverDestroyed<const AtomString> name("month"_s);
    return name;
}

const AtomString& number()
{
    static MainThreadNeverDestroyed<const AtomString> name("number"_s);
    return name;
}

const AtomString& password()
{
    static MainThreadNeverDestroyed<const AtomString> name("password"_s);
    return name;
}

const AtomString& radio()
{
    static MainThreadNeverDestroyed<const AtomString> name("radio"_s);
    return name;
}

const AtomString& range()
{
    static MainThreadNeverDestroyed<const AtomString> name("range"_s);
    return name;
}

const AtomString& reset()
{
    return resetAtom();
}

const AtomString& search()
{
    return searchAtom();
}

const AtomString& submit()
{
    return submitAtom();
}

const AtomString& telephone()
{
    return telAtom();
}

const AtomString& text()
{
    return textAtom();
}

const AtomString& time()
{
    static MainThreadNeverDestroyed<const AtomString> name("time"_s);
    return name;
}

const AtomString& url()
{
    return urlAtom();
}

const AtomString& week()
{
    static MainThreadNeverDestroyed<const AtomString> name("week"_s);
    return name;
}

} // namespace WebCore::InputTypeNames

} // namespace WebCore
