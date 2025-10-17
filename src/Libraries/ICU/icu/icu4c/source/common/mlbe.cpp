/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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

// Â© 2022 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#if !UCONFIG_NO_BREAK_ITERATION

#include "cmemory.h"
#include "mlbe.h"
#include "uassert.h"
#include "ubrkimpl.h"
#include "unicode/resbund.h"
#include "unicode/udata.h"
#include "unicode/utf16.h"
#include "uresimp.h"
#include "util.h"
#include "uvectr32.h"

U_NAMESPACE_BEGIN

enum class ModelIndex { kUWStart = 0, kBWStart = 6, kTWStart = 9 };

MlBreakEngine::MlBreakEngine(const UnicodeSet &digitOrOpenPunctuationOrAlphabetSet,
                             const UnicodeSet &closePunctuationSet, UErrorCode &status)
    : fDigitOrOpenPunctuationOrAlphabetSet(digitOrOpenPunctuationOrAlphabetSet),
      fClosePunctuationSet(closePunctuationSet),
      fNegativeSum(0) {
    if (U_FAILURE(status)) {
        return;
    }
    loadMLModel(status);
}

MlBreakEngine::~MlBreakEngine() {}

int32_t MlBreakEngine::divideUpRange(UText *inText, int32_t rangeStart, int32_t rangeEnd,
                                     UVector32 &foundBreaks, const UnicodeString &inString,
                                     const LocalPointer<UVector32> &inputMap,
                                     UErrorCode &status) const {
    if (U_FAILURE(status)) {
        return 0;
    }
    if (rangeStart >= rangeEnd) {
        status = U_ILLEGAL_ARGUMENT_ERROR;
        return 0;
    }

    UVector32 boundary(inString.countChar32() + 1, status);
    if (U_FAILURE(status)) {
        return 0;
    }
    int32_t numBreaks = 0;
    int32_t codePointLength = inString.countChar32();
    // The ML algorithm groups six char and evaluates whether the 4th char is a breakpoint.
    // In each iteration, it evaluates the 4th char and then moves forward one char like a sliding
    // window. Initially, the first six values in the indexList are [-1, -1, 0, 1, 2, 3]. After
    // moving forward, finally the last six values in the indexList are
    // [length-4, length-3, length-2, length-1, -1, -1]. The "+4" here means four extra "-1".
    int32_t indexSize = codePointLength + 4;
    LocalMemory<int32_t> indexList(static_cast<int32_t*>(uprv_malloc(indexSize * sizeof(int32_t))));
    if (indexList.isNull()) {
        status = U_MEMORY_ALLOCATION_ERROR;
        return 0;
    }
    int32_t numCodeUnits = initIndexList(inString, indexList.getAlias(), status);

    // Add a break for the start.
    boundary.addElement(0, status);
    numBreaks++;
    if (U_FAILURE(status)) return 0;

    for (int32_t idx = 0; idx + 1 < codePointLength && U_SUCCESS(status); idx++) {
        numBreaks =
            evaluateBreakpoint(inString, indexList.getAlias(), idx, numCodeUnits, numBreaks, boundary, status);
        if (idx + 4 < codePointLength) {
            indexList[idx + 6] = numCodeUnits;
            numCodeUnits += U16_LENGTH(inString.char32At(indexList[idx + 6]));
        }
    }

    if (U_FAILURE(status)) return 0;

    // Add a break for the end if there is not one there already.
    if (boundary.lastElementi() != inString.countChar32()) {
        boundary.addElement(inString.countChar32(), status);
        numBreaks++;
    }

    int32_t prevCPPos = -1;
    int32_t prevUTextPos = -1;
    int32_t correctedNumBreaks = 0;
    for (int32_t i = 0; i < numBreaks; i++) {
        int32_t cpPos = boundary.elementAti(i);
        int32_t utextPos = inputMap.isValid() ? inputMap->elementAti(cpPos) : cpPos + rangeStart;
        U_ASSERT(cpPos > prevCPPos);
        U_ASSERT(utextPos >= prevUTextPos);

        if (utextPos > prevUTextPos) {
            if (utextPos != rangeStart ||
                (utextPos > 0 &&
                 fClosePunctuationSet.contains(utext_char32At(inText, utextPos - 1)))) {
                foundBreaks.push(utextPos, status);
                correctedNumBreaks++;
            }
        } else {
            // Normalization expanded the input text, the dictionary found a boundary
            // within the expansion, giving two boundaries with the same index in the
            // original text. Ignore the second. See ticket #12918.
            --numBreaks;
        }
        prevCPPos = cpPos;
        prevUTextPos = utextPos;
    }
    (void)prevCPPos;  // suppress compiler warnings about unused variable

    UChar32 nextChar = utext_char32At(inText, rangeEnd);
    if (!foundBreaks.isEmpty() && foundBreaks.peeki() == rangeEnd) {
        // In phrase breaking, there has to be a breakpoint between Cj character and
        // the number/open punctuation.
        // E.g. ã‚‹æ–‡å­—ã€Œãã†ã ã€äº¬éƒ½ã€->ã‚‹â–æ–‡å­—â–ã€Œãã†ã ã€â–äº¬éƒ½ã€-> breakpoint between å­— andã€Œ
        // E.g. ä¹—è»ŠçŽ‡ï¼™ï¼ï¼…ç¨‹åº¦ã ã‚ã†ã‹ -> ä¹—è»Šâ–çŽ‡â–ï¼™ï¼ï¼…â–ç¨‹åº¦ã ã‚ã†ã‹ -> breakpoint between çŽ‡ and ï¼™
        // E.g. ã—ã‹ã‚‚ãƒ­ã‚´ãŒï¼µï½Žï½‰ï½ƒï½ï½„ï½…ï¼ -> ã—ã‹ã‚‚â–ãƒ­ã‚´ãŒâ–ï¼µï½Žï½‰ï½ƒï½ï½„ï½…ï¼-> breakpoint between ãŒ and ï¼µ
        if (!fDigitOrOpenPunctuationOrAlphabetSet.contains(nextChar)) {
            foundBreaks.popi();
            correctedNumBreaks--;
        }
    }

    return correctedNumBreaks;
}

int32_t MlBreakEngine::evaluateBreakpoint(const UnicodeString &inString, int32_t *indexList,
                                          int32_t startIdx, int32_t numCodeUnits, int32_t numBreaks,
                                          UVector32 &boundary, UErrorCode &status) const {
    if (U_FAILURE(status)) {
        return numBreaks;
    }
    int32_t start = 0, end = 0;
    int32_t score = fNegativeSum;

    for (int i = 0; i < 6; i++) {
        // UW1 ~ UW6
        start = startIdx + i;
        if (indexList[start] != -1) {
            end = (indexList[start + 1] != -1) ? indexList[start + 1] : numCodeUnits;
            score += fModel[static_cast<int32_t>(ModelIndex::kUWStart) + i].geti(
                inString.tempSubString(indexList[start], end - indexList[start]));
        }
    }
    for (int i = 0; i < 3; i++) {
        // BW1 ~ BW3
        start = startIdx + i + 1;
        if (indexList[start] != -1 && indexList[start + 1] != -1) {
            end = (indexList[start + 2] != -1) ? indexList[start + 2] : numCodeUnits;
            score += fModel[static_cast<int32_t>(ModelIndex::kBWStart) + i].geti(
                inString.tempSubString(indexList[start], end - indexList[start]));
        }
    }
    for (int i = 0; i < 4; i++) {
        // TW1 ~ TW4
        start = startIdx + i;
        if (indexList[start] != -1 && indexList[start + 1] != -1 && indexList[start + 2] != -1) {
            end = (indexList[start + 3] != -1) ? indexList[start + 3] : numCodeUnits;
            score += fModel[static_cast<int32_t>(ModelIndex::kTWStart) + i].geti(
                inString.tempSubString(indexList[start], end - indexList[start]));
        }
    }

    if (score > 0) {
        boundary.addElement(startIdx + 1, status);
        numBreaks++;
    }
    return numBreaks;
}

int32_t MlBreakEngine::initIndexList(const UnicodeString &inString, int32_t *indexList,
                                     UErrorCode &status) const {
    if (U_FAILURE(status)) {
        return 0;
    }
    int32_t index = 0;
    int32_t length = inString.countChar32();
    // Set all (lenght+4) items inside indexLength to -1 presuming -1 is 4 bytes of 0xff.
    uprv_memset(indexList, 0xff, (length + 4) * sizeof(int32_t));
    if (length > 0) {
        indexList[2] = 0;
        index = U16_LENGTH(inString.char32At(0));
        if (length > 1) {
            indexList[3] = index;
            index += U16_LENGTH(inString.char32At(index));
            if (length > 2) {
                indexList[4] = index;
                index += U16_LENGTH(inString.char32At(index));
                if (length > 3) {
                    indexList[5] = index;
                    index += U16_LENGTH(inString.char32At(index));
                }
            }
        }
    }
    return index;
}

void MlBreakEngine::loadMLModel(UErrorCode &error) {
    // BudouX's model consists of thirteen categories, each of which is make up of pairs of the
    // feature and its score. As integrating it into jaml.txt, we define thirteen kinds of key and
    // value to represent the feature and the corresponding score respectively.

    if (U_FAILURE(error)) return;

    UnicodeString key;
    StackUResourceBundle stackTempBundle;
    ResourceDataValue modelKey;

    LocalUResourceBundlePointer rbp(ures_openDirect(U_ICUDATA_BRKITR, "jaml", &error));
    UResourceBundle *rb = rbp.getAlias();
    if (U_FAILURE(error)) return;

    int32_t index = 0;
    initKeyValue(rb, "UW1Keys", "UW1Values", fModel[index++], error);
    initKeyValue(rb, "UW2Keys", "UW2Values", fModel[index++], error);
    initKeyValue(rb, "UW3Keys", "UW3Values", fModel[index++], error);
    initKeyValue(rb, "UW4Keys", "UW4Values", fModel[index++], error);
    initKeyValue(rb, "UW5Keys", "UW5Values", fModel[index++], error);
    initKeyValue(rb, "UW6Keys", "UW6Values", fModel[index++], error);
    initKeyValue(rb, "BW1Keys", "BW1Values", fModel[index++], error);
    initKeyValue(rb, "BW2Keys", "BW2Values", fModel[index++], error);
    initKeyValue(rb, "BW3Keys", "BW3Values", fModel[index++], error);
    initKeyValue(rb, "TW1Keys", "TW1Values", fModel[index++], error);
    initKeyValue(rb, "TW2Keys", "TW2Values", fModel[index++], error);
    initKeyValue(rb, "TW3Keys", "TW3Values", fModel[index++], error);
    initKeyValue(rb, "TW4Keys", "TW4Values", fModel[index++], error);
    fNegativeSum /= 2;
}

void MlBreakEngine::initKeyValue(UResourceBundle *rb, const char *keyName, const char *valueName,
                                 Hashtable &model, UErrorCode &error) {
    int32_t keySize = 0;
    int32_t valueSize = 0;
    int32_t stringLength = 0;
    UnicodeString key;
    StackUResourceBundle stackTempBundle;
    ResourceDataValue modelKey;

    // get modelValues
    LocalUResourceBundlePointer modelValue(ures_getByKey(rb, valueName, nullptr, &error));
    const int32_t *value = ures_getIntVector(modelValue.getAlias(), &valueSize, &error);
    if (U_FAILURE(error)) return;

    // get modelKeys
    ures_getValueWithFallback(rb, keyName, stackTempBundle.getAlias(), modelKey, error);
    ResourceArray stringArray = modelKey.getArray(error);
    keySize = stringArray.getSize();
    if (U_FAILURE(error)) return;

    for (int32_t idx = 0; idx < keySize; idx++) {
        stringArray.getValue(idx, modelKey);
        key = UnicodeString(modelKey.getString(stringLength, error));
        if (U_SUCCESS(error)) {
            U_ASSERT(idx < valueSize);
            fNegativeSum -= value[idx];
            model.puti(key, value[idx], error);
        }
    }
}

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_BREAK_ITERATION */
