/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
#include <wtf/text/TextBreakIteratorInternalICU.h>

namespace WTF {

const char* currentSearchLocaleID()
{
    // FIXME: Should use system locale.
    return "";
}

const char* currentTextBreakLocaleID()
{
    // Using en_US_POSIX now so word selection in address field works as expected as before (double-clicking
    // in a URL selects a word delimited by periods rather than selecting the entire URL).
    // However, this is not entirely correct - we should honor the system locale in the normal case.
    // FIXME: <rdar://problem/6786703> Should use system locale for text breaking
    return "en_US_POSIX";
}

}
