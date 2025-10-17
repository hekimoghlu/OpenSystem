/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
 * Copyright (C) 2015, International Business Machines Corporation and
 * others. All Rights Reserved.
 */

#include "unicode/unistr.h"
#include "charstr.h"
#include "cstring.h"
#include "pluralmap.h"

U_NAMESPACE_BEGIN

static const char * const gPluralForms[] = {
        "other", "zero", "one", "two", "few", "many"};

PluralMapBase::Category
PluralMapBase::toCategory(const char *pluralForm) {
    for (int32_t i = 0; i < UPRV_LENGTHOF(gPluralForms); ++i) {
        if (uprv_strcmp(pluralForm, gPluralForms[i]) == 0) {
            return static_cast<Category>(i);
        }
    }
    return NONE;
}

PluralMapBase::Category
PluralMapBase::toCategory(const UnicodeString &pluralForm) {
    CharString cCategory;
    UErrorCode status = U_ZERO_ERROR;
    cCategory.appendInvariantChars(pluralForm, status);    
    return U_FAILURE(status) ? NONE : toCategory(cCategory.data());
}

const char *PluralMapBase::getCategoryName(Category c) {
    int32_t index = c;
    return (index < 0 || index >= UPRV_LENGTHOF(gPluralForms)) ?
            nullptr : gPluralForms[index];
}


U_NAMESPACE_END

