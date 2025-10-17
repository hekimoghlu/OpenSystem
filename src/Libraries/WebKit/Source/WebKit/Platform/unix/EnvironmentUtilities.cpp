/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
#include "EnvironmentUtilities.h"

#include <cstdlib>
#include <wtf/text/StringBuilder.h>

namespace WebKit {

namespace EnvironmentUtilities {

String stripEntriesEndingWith(StringView input, StringView suffix)
{
    StringBuilder output;

    auto hasAppended = false;
    for (auto entry : input.splitAllowingEmptyEntries(':')) {
        if (entry.endsWith(suffix))
            continue;

        if (hasAppended)
            output.append(':');
        else
            hasAppended = true;

        output.append(entry);
    }

    return output.toString();
}

void removeValuesEndingWith(const char* environmentVariable, const char* searchValue)
{
    const char* before = getenv(environmentVariable);
    if (!before)
        return;

    auto after = stripEntriesEndingWith(StringView::fromLatin1(before), StringView::fromLatin1(searchValue));
    if (after.isEmpty()) {
        unsetenv(environmentVariable);
        return;
    }

    setenv(environmentVariable, after.utf8().data(), 1);
}

} // namespace EnvironmentUtilities

} // namespace WebKit
