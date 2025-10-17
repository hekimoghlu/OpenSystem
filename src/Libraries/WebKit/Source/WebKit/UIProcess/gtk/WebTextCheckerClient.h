/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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

#include "APIClient.h"
#include "WKTextCheckerGLib.h"
#include <WebCore/TextCheckerClient.h>
#include <wtf/Forward.h>

namespace API {
template<> struct ClientTraits<WKTextCheckerClientBase> {
    typedef std::tuple<WKTextCheckerClientV0> Versions;
};
}

namespace WebKit {

class WebPageProxy;

class WebTextCheckerClient : public API::Client<WKTextCheckerClientBase> {
public:
    bool continuousSpellCheckingAllowed();
    bool continuousSpellCheckingEnabled();
    void setContinuousSpellCheckingEnabled(bool);
    bool grammarCheckingEnabled();
    void setGrammarCheckingEnabled(bool);
    uint64_t uniqueSpellDocumentTag(WebPageProxy*);
    void closeSpellDocumentWithTag(uint64_t);
    void checkSpellingOfString(uint64_t tag, const String& text, int32_t& misspellingLocation, int32_t& misspellingLength);
    void checkGrammarOfString(uint64_t tag, const String& text, Vector<WebCore::GrammarDetail>&, int32_t& badGrammarLocation, int32_t& badGrammarLength);
    bool spellingUIIsShowing();
    void toggleSpellingUIIsShowing();
    void updateSpellingUIWithMisspelledWord(uint64_t tag, const String& misspelledWord);
    void updateSpellingUIWithGrammarString(uint64_t tag, const String& badGrammarPhrase, const WebCore::GrammarDetail&);
    void guessesForWord(uint64_t tag, const String& word, Vector<String>& guesses);
    void learnWord(uint64_t tag, const String& word);
    void ignoreWord(uint64_t tag, const String& word);
};

} // namespace WebKit
