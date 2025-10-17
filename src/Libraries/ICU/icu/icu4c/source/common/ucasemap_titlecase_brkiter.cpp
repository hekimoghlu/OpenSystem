/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
*******************************************************************************
*   Copyright (C) 2011, International Business Machines
*   Corporation and others.  All Rights Reserved.
*******************************************************************************
*   file name:  ucasemap_titlecase_brkiter.cpp
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2011jun02
*   created by: Markus W. Scherer
*
*   Titlecasing functions that are based on BreakIterator
*   were moved here to break dependency cycles among parts of the common library.
*/

#include "unicode/utypes.h"

#if !UCONFIG_NO_BREAK_ITERATION

#include "unicode/brkiter.h"
#include "unicode/ubrk.h"
#include "unicode/casemap.h"
#include "unicode/ucasemap.h"
#include "cmemory.h"
#include "ucase.h"
#include "ucasemap_imp.h"

U_NAMESPACE_BEGIN

void CaseMap::utf8ToTitle(
        const char *locale, uint32_t options, BreakIterator *iter,
        StringPiece src, ByteSink &sink, Edits *edits,
        UErrorCode &errorCode) {
    if (U_FAILURE(errorCode)) {
        return;
    }
    UText utext = UTEXT_INITIALIZER;
    utext_openUTF8(&utext, src.data(), src.length(), &errorCode);
    LocalPointer<BreakIterator> ownedIter;
    iter = ustrcase_getTitleBreakIterator(nullptr, locale, options, iter, ownedIter, errorCode);
    if (iter == nullptr) {
        utext_close(&utext);
        return;
    }
    iter->setText(&utext, errorCode);
    ucasemap_mapUTF8(
        ustrcase_getCaseLocale(locale), options, iter,
        src.data(), src.length(),
        ucasemap_internalUTF8ToTitle, sink, edits, errorCode);
    utext_close(&utext);
}

int32_t CaseMap::utf8ToTitle(
        const char *locale, uint32_t options, BreakIterator *iter,
        const char *src, int32_t srcLength,
        char *dest, int32_t destCapacity, Edits *edits,
        UErrorCode &errorCode) {
    if (U_FAILURE(errorCode)) {
        return 0;
    }
    UText utext=UTEXT_INITIALIZER;
    utext_openUTF8(&utext, src, srcLength, &errorCode);
    LocalPointer<BreakIterator> ownedIter;
    iter = ustrcase_getTitleBreakIterator(nullptr, locale, options, iter, ownedIter, errorCode);
    if(iter==nullptr) {
        utext_close(&utext);
        return 0;
    }
    iter->setText(&utext, errorCode);
    int32_t length=ucasemap_mapUTF8(
        ustrcase_getCaseLocale(locale), options, iter,
        dest, destCapacity,
        src, srcLength,
        ucasemap_internalUTF8ToTitle, edits, errorCode);
    utext_close(&utext);
    return length;
}

U_NAMESPACE_END

U_NAMESPACE_USE

U_CAPI const UBreakIterator * U_EXPORT2
ucasemap_getBreakIterator(const UCaseMap *csm) {
    return reinterpret_cast<UBreakIterator *>(csm->iter);
}

U_CAPI void U_EXPORT2
ucasemap_setBreakIterator(UCaseMap *csm, UBreakIterator *iterToAdopt, UErrorCode *pErrorCode) {
    if(U_FAILURE(*pErrorCode)) {
        return;
    }
    delete csm->iter;
    csm->iter=reinterpret_cast<BreakIterator *>(iterToAdopt);
}

U_CAPI int32_t U_EXPORT2
ucasemap_utf8ToTitle(UCaseMap *csm,
                     char *dest, int32_t destCapacity,
                     const char *src, int32_t srcLength,
                     UErrorCode *pErrorCode) {
    if (U_FAILURE(*pErrorCode)) {
        return 0;
    }
    UText utext=UTEXT_INITIALIZER;
    utext_openUTF8(&utext, src, srcLength, pErrorCode);
    if (U_FAILURE(*pErrorCode)) {
        return 0;
    }
    if(csm->iter==nullptr) {
        LocalPointer<BreakIterator> ownedIter;
        BreakIterator *iter = ustrcase_getTitleBreakIterator(
            nullptr, csm->locale, csm->options, nullptr, ownedIter, *pErrorCode);
        if (iter == nullptr) {
            utext_close(&utext);
            return 0;
        }
        csm->iter = ownedIter.orphan();
    }
    csm->iter->setText(&utext, *pErrorCode);
    int32_t length=ucasemap_mapUTF8(
            csm->caseLocale, csm->options, csm->iter,
            dest, destCapacity,
            src, srcLength,
            ucasemap_internalUTF8ToTitle, nullptr, *pErrorCode);
    utext_close(&utext);
    return length;
}

#endif  // !UCONFIG_NO_BREAK_ITERATION
