/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
#include "URLInputType.h"

#include "HTMLInputElement.h"
#include "InputTypeNames.h"
#include "LocalizedStrings.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(URLInputType);

const AtomString& URLInputType::formControlType() const
{
    return InputTypeNames::url();
}

bool URLInputType::typeMismatchFor(const String& value) const
{
    return !value.isEmpty() && !URL(value).isValid();
}

bool URLInputType::typeMismatch() const
{
    ASSERT(element());
    return typeMismatchFor(element()->value());
}

String URLInputType::typeMismatchText() const
{
    return validationMessageTypeMismatchForURLText();
}

String URLInputType::sanitizeValue(const String& proposedValue) const
{
    return BaseTextInputType::sanitizeValue(proposedValue).trim(isASCIIWhitespace);
}

} // namespace WebCore
