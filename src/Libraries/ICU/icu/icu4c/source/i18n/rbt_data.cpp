/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
**********************************************************************
*   Copyright (C) 1999-2014, International Business Machines
*   Corporation and others.  All Rights Reserved.
**********************************************************************
*   Date        Name        Description
*   11/17/99    aliu        Creation.
**********************************************************************
*/

#include "unicode/utypes.h"
#include "umutex.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "unicode/unistr.h"
#include "unicode/uniset.h"
#include "rbt_data.h"
#include "hash.h"
#include "cmemory.h"

U_NAMESPACE_BEGIN

TransliterationRuleData::TransliterationRuleData(UErrorCode& status)
 : UMemory(), ruleSet(status), variableNames(status),
    variables(nullptr), variablesAreOwned(true)
{
    if (U_FAILURE(status)) {
        return;
    }
    variableNames.setValueDeleter(uprv_deleteUObject);
    variables = nullptr;
    variablesLength = 0;
}

TransliterationRuleData::TransliterationRuleData(const TransliterationRuleData& other) :
    UMemory(other), ruleSet(other.ruleSet),
    variablesAreOwned(true),
    variablesBase(other.variablesBase),
    variablesLength(other.variablesLength)
{
    UErrorCode status = U_ZERO_ERROR;
    int32_t i = 0;
    variableNames.setValueDeleter(uprv_deleteUObject);
    int32_t pos = UHASH_FIRST;
    const UHashElement *e;
    while ((e = other.variableNames.nextElement(pos)) != nullptr) {
        UnicodeString* value =
            new UnicodeString(*static_cast<const UnicodeString*>(e->value.pointer));
        // Exit out if value could not be created.
        if (value == nullptr) {
        	return;
        }
        variableNames.put(*static_cast<UnicodeString*>(e->key.pointer), value, status);
    }

    variables = nullptr;
    if (other.variables != nullptr) {
        variables = static_cast<UnicodeFunctor**>(uprv_malloc(variablesLength * sizeof(UnicodeFunctor*)));
        /* test for nullptr */
        if (variables == nullptr) {
            status = U_MEMORY_ALLOCATION_ERROR;
            return;
        }
        for (i=0; i<variablesLength; ++i) {
            variables[i] = other.variables[i]->clone();
            if (variables[i] == nullptr) {
                status = U_MEMORY_ALLOCATION_ERROR;
                break;
            }
        }
    }
    // Remove the array and exit if memory allocation error occurred.
    if (U_FAILURE(status)) {
        for (int32_t n = i-1; n >= 0; n--) {
            delete variables[n];
        }
        uprv_free(variables);
        variables = nullptr;
        return;
    }

    // Do this last, _after_ setting up variables[].
    ruleSet.setData(this); // ruleSet must already be frozen
}

TransliterationRuleData::~TransliterationRuleData() {
    if (variablesAreOwned && variables != nullptr) {
        for (int32_t i=0; i<variablesLength; ++i) {
            delete variables[i];
        }
    }
    uprv_free(variables);
}

UnicodeFunctor*
TransliterationRuleData::lookup(UChar32 standIn) const {
    int32_t i = standIn - variablesBase;
    return (i >= 0 && i < variablesLength) ? variables[i] : nullptr;
}

UnicodeMatcher*
TransliterationRuleData::lookupMatcher(UChar32 standIn) const {
    UnicodeFunctor *f = lookup(standIn);
    return f != nullptr ? f->toMatcher() : nullptr;
}

UnicodeReplacer*
TransliterationRuleData::lookupReplacer(UChar32 standIn) const {
    UnicodeFunctor *f = lookup(standIn);
    return f != nullptr ? f->toReplacer() : nullptr;
}


U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_TRANSLITERATION */
