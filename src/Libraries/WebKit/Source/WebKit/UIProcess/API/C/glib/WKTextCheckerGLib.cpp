/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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
#include "WKTextCheckerGLib.h"

#include "TextChecker.h"
#include "WKAPICast.h"
#include "WebPageProxy.h"

#if PLATFORM(GTK)
#include "WebTextChecker.h"
#endif

using namespace WebKit;

#if PLATFORM(GTK)

void WKTextCheckerSetClient(const WKTextCheckerClientBase* wkClient)
{
    if (wkClient && wkClient->version)
        return;
    WebTextChecker::singleton()->setClient(wkClient);
}

void WKTextCheckerContinuousSpellCheckingEnabledStateChanged(bool enabled)
{
    WebTextChecker::singleton()->continuousSpellCheckingEnabledStateChanged(enabled);
}

void WKTextCheckerGrammarCheckingEnabledStateChanged(bool enabled)
{
    WebTextChecker::singleton()->grammarCheckingEnabledStateChanged(enabled);
}

void WKTextCheckerCheckSpelling(WKPageRef page, bool startBeforeSelection)
{
    WebTextChecker::singleton()->checkSpelling(toImpl(page), startBeforeSelection);
}

void WKTextCheckerChangeSpellingToWord(WKPageRef page, WKStringRef word)
{
    WebTextChecker::singleton()->changeSpellingToWord(toImpl(page), toWTFString(word));
}

void WKTextCheckerSetSpellCheckingLanguages(const char* const* languages)
{
#if ENABLE(SPELLCHECK)
    Vector<String> spellCheckingLanguages;
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GTK/WPE port
    for (size_t i = 0; languages[i]; ++i)
        spellCheckingLanguages.append(String::fromUTF8(languages[i]));
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    WebKit::TextChecker::setSpellCheckingLanguages(spellCheckingLanguages);
#endif
}

#endif // PLATFORM(GTK)

void WKTextCheckerSetContinuousSpellCheckingEnabled(bool enabled)
{
    WebKit::TextChecker::setContinuousSpellCheckingEnabled(enabled);
}
