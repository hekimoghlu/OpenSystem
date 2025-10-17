/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
#include "WebTextChecker.h"

#include "TextChecker.h"
#include "WKAPICast.h"
#include "WebPageProxy.h"
#include "WebProcessPool.h"
#include <wtf/RefPtr.h>

namespace WebKit {

WebTextChecker* WebTextChecker::singleton()
{
    static WebTextChecker* textChecker = adoptRef(new WebTextChecker).leakRef();
    return textChecker;
}

WebTextChecker::WebTextChecker()
{
}

void WebTextChecker::setClient(const WKTextCheckerClientBase* client)
{
    m_client.initialize(client);
}

static void updateStateForAllContexts()
{
    for (auto& processPool : WebProcessPool::allProcessPools())
        processPool->textCheckerStateChanged();
}

void WebTextChecker::continuousSpellCheckingEnabledStateChanged(bool enabled)
{
    TextChecker::continuousSpellCheckingEnabledStateChanged(enabled);
    updateStateForAllContexts();
}

void WebTextChecker::grammarCheckingEnabledStateChanged(bool enabled)
{
    TextChecker::grammarCheckingEnabledStateChanged(enabled);
    updateStateForAllContexts();
}

void WebTextChecker::checkSpelling(WebPageProxy* page, bool startBeforeSelection)
{
    page->advanceToNextMisspelling(startBeforeSelection);
}

void WebTextChecker::changeSpellingToWord(WebPageProxy* page, const String& text)
{
    page->changeSpellingToWord(text);
}

} // namespace WebKit
