/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#pragma once

#include <WebKit/WKBase.h>
#include <WebKit/WKTextChecker.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT void WKTextCheckerSetTestingMode(bool enabled);

#if PLATFORM(GTK)

// TextChecker Client
typedef bool (*WKTextCheckerContinousSpellCheckingAllowed)(const void *clientInfo);
typedef bool (*WKTextCheckerContinousSpellCheckingEnabled)(const void *clientInfo);
typedef void (*WKTextCheckerSetContinousSpellCheckingEnabled)(bool enabled, const void *clientInfo);
typedef bool (*WKTextCheckerGrammarCheckingEnabled)(const void *clientInfo);
typedef void (*WKTextCheckerSetGrammarCheckingEnabled)(bool enabled, const void *clientInfo);
typedef uint64_t (*WKTextCheckerUniqueSpellDocumentTag)(WKPageRef page, const void *clientInfo);
typedef void (*WKTextCheckerCloseSpellDocumentWithTag)(uint64_t tag, const void *clientInfo);
typedef void (*WKTextCheckerCheckSpellingOfString)(uint64_t tag, WKStringRef text, int32_t* misspellingLocation, int32_t* misspellingLength, const void *clientInfo);
typedef void (*WKTextCheckerCheckGrammarOfString)(uint64_t tag, WKStringRef text, WKArrayRef* grammarDetails, int32_t* badGrammarLocation, int32_t* badGrammarLength, const void *clientInfo);
typedef bool (*WKTextCheckerSpellingUIIsShowing)(const void *clientInfo);
typedef void (*WKTextCheckerToggleSpellingUIIsShowing)(const void *clientInfo);
typedef void (*WKTextCheckerUpdateSpellingUIWithMisspelledWord)(uint64_t tag, WKStringRef misspelledWord, const void *clientInfo);
typedef void (*WKTextCheckerUpdateSpellingUIWithGrammarString)(uint64_t tag, WKStringRef badGrammarPhrase, WKGrammarDetailRef grammarDetail, const void *clientInfo);
typedef WKArrayRef (*WKTextCheckerGuessesForWord)(uint64_t tag, WKStringRef word, const void *clientInfo);
typedef void (*WKTextCheckerLearnWord)(uint64_t tag, WKStringRef word, const void *clientInfo);
typedef void (*WKTextCheckerIgnoreWord)(uint64_t tag, WKStringRef word, const void *clientInfo);

typedef struct WKTextCheckerClientBase {
    int                                                                     version;
    const void *                                                            clientInfo;
} WKTextCheckerClientBase;

typedef struct WKTextCheckerClientV0 {
    WKTextCheckerClientBase                                                 base;

    WKTextCheckerContinousSpellCheckingAllowed                              continuousSpellCheckingAllowed;
    WKTextCheckerContinousSpellCheckingEnabled                              continuousSpellCheckingEnabled;
    WKTextCheckerSetContinousSpellCheckingEnabled                           setContinuousSpellCheckingEnabled;
    WKTextCheckerGrammarCheckingEnabled                                     grammarCheckingEnabled;
    WKTextCheckerSetGrammarCheckingEnabled                                  setGrammarCheckingEnabled;
    WKTextCheckerUniqueSpellDocumentTag                                     uniqueSpellDocumentTag;
    WKTextCheckerCloseSpellDocumentWithTag                                  closeSpellDocumentWithTag;
    WKTextCheckerCheckSpellingOfString                                      checkSpellingOfString;
    WKTextCheckerCheckGrammarOfString                                       checkGrammarOfString;
    WKTextCheckerSpellingUIIsShowing                                        spellingUIIsShowing;
    WKTextCheckerToggleSpellingUIIsShowing                                  toggleSpellingUIIsShowing;
    WKTextCheckerUpdateSpellingUIWithMisspelledWord                         updateSpellingUIWithMisspelledWord;
    WKTextCheckerUpdateSpellingUIWithGrammarString                          updateSpellingUIWithGrammarString;
    WKTextCheckerGuessesForWord                                             guessesForWord;
    WKTextCheckerLearnWord                                                  learnWord;
    WKTextCheckerIgnoreWord                                                 ignoreWord;
} WKTextCheckerClientV0;

WK_EXPORT void WKTextCheckerSetClient(const WKTextCheckerClientBase* client);

WK_EXPORT void WKTextCheckerContinuousSpellCheckingEnabledStateChanged(bool);
WK_EXPORT void WKTextCheckerGrammarCheckingEnabledStateChanged(bool);

WK_EXPORT void WKTextCheckerCheckSpelling(WKPageRef page, bool startBeforeSelection);
WK_EXPORT void WKTextCheckerChangeSpellingToWord(WKPageRef page, WKStringRef word);

WK_EXPORT void WKTextCheckerSetSpellCheckingLanguages(const char* const* languages);

#endif // PLATFORM(GTK)

WK_EXPORT void WKTextCheckerSetContinuousSpellCheckingEnabled(bool);

#ifdef __cplusplus
}
#endif
