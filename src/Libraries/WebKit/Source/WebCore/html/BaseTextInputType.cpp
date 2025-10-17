/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
#include "BaseTextInputType.h"

#include "ElementInlines.h"
#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include <JavaScriptCore/RegularExpression.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BaseTextInputType);

using namespace HTMLNames;

bool BaseTextInputType::patternMismatch(const String& value) const
{
    ASSERT(element());
    // FIXME: We should execute RegExp parser first to check validity instead of creating an actual RegularExpression.
    // https://bugs.webkit.org/show_bug.cgi?id=183361
    const AtomString& rawPattern = element()->attributeWithoutSynchronization(patternAttr);
    if (rawPattern.isNull() || value.isEmpty() || !JSC::Yarr::RegularExpression(rawPattern, { JSC::Yarr::Flags::UnicodeSets }).isValid())
        return false;

    String pattern = makeString("^(?:"_s, rawPattern, ")$"_s);
    JSC::Yarr::RegularExpression regex(pattern, { JSC::Yarr::Flags::UnicodeSets });
    auto valuePatternMismatch = [&regex](auto& value) {
        int matchLength = 0;
        int valueLength = value.length();
        int matchOffset = regex.match(value, 0, &matchLength);
        return matchOffset || matchLength != valueLength;
    };

    if (isEmailField() && element()->multiple()) {
        auto values = value.split(',');
        return values.findIf(valuePatternMismatch) != notFound;
    }
    return valuePatternMismatch(value);
}

bool BaseTextInputType::supportsPlaceholder() const
{
    return true;
}

bool BaseTextInputType::supportsSelectionAPI() const
{
    return true;
}

bool BaseTextInputType::dirAutoUsesValue() const
{
    return true;
}

} // namespace WebCore
