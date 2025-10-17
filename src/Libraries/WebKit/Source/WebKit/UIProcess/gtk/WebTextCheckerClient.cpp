/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#include "WebTextCheckerClient.h"

#include "APIArray.h"
#include "WKAPICast.h"
#include "WKSharedAPICast.h"
#include "WebGrammarDetail.h"
#include "WebPageProxy.h"
#include <wtf/text/WTFString.h>

namespace WebKit {

bool WebTextCheckerClient::continuousSpellCheckingAllowed()
{
    if (!m_client.continuousSpellCheckingAllowed)
        return false;

    return m_client.continuousSpellCheckingAllowed(m_client.base.clientInfo);
}

bool WebTextCheckerClient::continuousSpellCheckingEnabled()
{
    if (!m_client.continuousSpellCheckingEnabled)
        return false;

    return m_client.continuousSpellCheckingEnabled(m_client.base.clientInfo);
}

void WebTextCheckerClient::setContinuousSpellCheckingEnabled(bool enabled)
{
    if (!m_client.setContinuousSpellCheckingEnabled)
        return;

    m_client.setContinuousSpellCheckingEnabled(enabled, m_client.base.clientInfo);
}

bool WebTextCheckerClient::grammarCheckingEnabled()
{
    if (!m_client.grammarCheckingEnabled)
        return false;

    return m_client.grammarCheckingEnabled(m_client.base.clientInfo);
}

void WebTextCheckerClient::setGrammarCheckingEnabled(bool enabled)
{
    if (!m_client.setGrammarCheckingEnabled)
        return;

    m_client.setGrammarCheckingEnabled(enabled, m_client.base.clientInfo);
}

uint64_t WebTextCheckerClient::uniqueSpellDocumentTag(WebPageProxy* page)
{
    if (!m_client.uniqueSpellDocumentTag)
        return 0;

    return m_client.uniqueSpellDocumentTag(toAPI(page), m_client.base.clientInfo);
}

void WebTextCheckerClient::closeSpellDocumentWithTag(uint64_t tag)
{
    if (!m_client.closeSpellDocumentWithTag)
        return;

    m_client.closeSpellDocumentWithTag(tag, m_client.base.clientInfo);
}

void WebTextCheckerClient::checkSpellingOfString(uint64_t tag, const String& text, int32_t& misspellingLocation, int32_t& misspellingLength)
{
    misspellingLocation = -1;
    misspellingLength = 0;

    if (!m_client.checkSpellingOfString)
        return;

    m_client.checkSpellingOfString(tag, toAPI(text.impl()), &misspellingLocation, &misspellingLength, m_client.base.clientInfo);
}

void WebTextCheckerClient::checkGrammarOfString(uint64_t tag, const String& text, Vector<WebCore::GrammarDetail>& grammarDetails, int32_t& badGrammarLocation, int32_t& badGrammarLength)
{
    badGrammarLocation = -1;
    badGrammarLength = 0;

    if (!m_client.checkGrammarOfString)
        return;

    WKArrayRef wkGrammarDetailsRef = 0;
    m_client.checkGrammarOfString(tag, toAPI(text.impl()), &wkGrammarDetailsRef, &badGrammarLocation, &badGrammarLength, m_client.base.clientInfo);

    RefPtr<API::Array> wkGrammarDetails = adoptRef(toImpl(wkGrammarDetailsRef));
    size_t numGrammarDetails = wkGrammarDetails->size();
    for (size_t i = 0; i < numGrammarDetails; ++i)
        grammarDetails.append(wkGrammarDetails->at<WebGrammarDetail>(i)->grammarDetail());
}

bool WebTextCheckerClient::spellingUIIsShowing()
{
    if (!m_client.spellingUIIsShowing)
        return false;

    return m_client.spellingUIIsShowing(m_client.base.clientInfo);
}

void WebTextCheckerClient::toggleSpellingUIIsShowing()
{
    if (!m_client.toggleSpellingUIIsShowing)
        return;

    return m_client.toggleSpellingUIIsShowing(m_client.base.clientInfo);
}

void WebTextCheckerClient::updateSpellingUIWithMisspelledWord(uint64_t tag, const String& misspelledWord)
{
    if (!m_client.updateSpellingUIWithMisspelledWord)
        return;

    m_client.updateSpellingUIWithMisspelledWord(tag, toAPI(misspelledWord.impl()), m_client.base.clientInfo);
}

void WebTextCheckerClient::updateSpellingUIWithGrammarString(uint64_t tag, const String& badGrammarPhrase, const WebCore::GrammarDetail& grammarDetail)
{
    if (!m_client.updateSpellingUIWithGrammarString)
        return;

    m_client.updateSpellingUIWithGrammarString(tag, toAPI(badGrammarPhrase.impl()), toAPI(grammarDetail), m_client.base.clientInfo);
}

void WebTextCheckerClient::guessesForWord(uint64_t tag, const String& word, Vector<String>& guesses)
{
    if (!m_client.guessesForWord)
        return;

    RefPtr<API::Array> wkGuesses = adoptRef(toImpl(m_client.guessesForWord(tag, toAPI(word.impl()), m_client.base.clientInfo)));
    size_t numGuesses = wkGuesses->size();
    for (size_t i = 0; i < numGuesses; ++i)
        guesses.append(wkGuesses->at<API::String>(i)->string());
}

void WebTextCheckerClient::learnWord(uint64_t tag, const String& word)
{
    if (!m_client.learnWord)
        return;

    m_client.learnWord(tag, toAPI(word.impl()), m_client.base.clientInfo);
}

void WebTextCheckerClient::ignoreWord(uint64_t tag, const String& word)
{
    if (!m_client.ignoreWord)
        return;

    m_client.ignoreWord(tag, toAPI(word.impl()), m_client.base.clientInfo);
}

} // namespace WebKit
