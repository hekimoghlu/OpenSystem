/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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
#include <wtf/Language.h>

#include <locale.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WTF {

// Using pango_language_get_default() here is not an option, because
// it doesn't support changing the locale in runtime, so it returns
// always the same value.
static String platformLanguage()
{
    auto localeDefault = String::fromLatin1(setlocale(LC_CTYPE, nullptr));
    if (localeDefault.isEmpty() || equalIgnoringASCIICase(localeDefault, "C"_s) || equalIgnoringASCIICase(localeDefault, "POSIX"_s))
        return "en-US"_s;

    auto normalizedDefault = makeStringByReplacingAll(localeDefault, '_', '-');
    return normalizedDefault.left(normalizedDefault.find('.'));
}

Vector<String> platformUserPreferredLanguages(ShouldMinimizeLanguages)
{
    return { platformLanguage() };
}

} // namespace WTF
