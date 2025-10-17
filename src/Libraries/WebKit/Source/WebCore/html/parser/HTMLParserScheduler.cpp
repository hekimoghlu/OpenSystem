/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#include "HTMLParserScheduler.h"

#include "Document.h"
#include "ElementInlines.h"
#include "HTMLDocumentParser.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "Page.h"
#include "ScriptController.h"
#include "ScriptElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(HTMLParserScheduler);

static Seconds parserTimeLimit(Page* page)
{
    // Always yield after exceeding this.
    constexpr auto defaultParserTimeLimit = 500_ms;

    // We're using the poorly named customHTMLTokenizerTimeDelay setting.
    if (page && page->hasCustomHTMLTokenizerTimeDelay())
        return Seconds(page->customHTMLTokenizerTimeDelay());
    return defaultParserTimeLimit;
}

ActiveParserSession::ActiveParserSession(Document* document)
    : m_document(document)
{
    if (!m_document)
        return;
    m_document->incrementActiveParserCount();
}

ActiveParserSession::~ActiveParserSession()
{
    if (!m_document)
        return;
    m_document->decrementActiveParserCount();
}

PumpSession::PumpSession(unsigned& nestingLevel, Document* document)
    : NestingLevelIncrementer(nestingLevel)
    , ActiveParserSession(document)
{
}

PumpSession::~PumpSession() = default;

Ref<HTMLParserScheduler> HTMLParserScheduler::create(HTMLDocumentParser& parser)
{
    return adoptRef(*new HTMLParserScheduler(parser));
}

HTMLParserScheduler::HTMLParserScheduler(HTMLDocumentParser& parser)
    : m_parser(&parser)
    , m_parserTimeLimit(parserTimeLimit(parser.document()->page()))
    , m_continueNextChunkTimer(*this, &HTMLParserScheduler::continueNextChunkTimerFired)
    , m_isSuspendedWithActiveTimer(false)
#if ASSERT_ENABLED
    , m_suspended(false)
#endif
{
}

HTMLParserScheduler::~HTMLParserScheduler() = default;

void HTMLParserScheduler::detach()
{
    m_continueNextChunkTimer.stop();
    m_parser = nullptr;
}

void HTMLParserScheduler::continueNextChunkTimerFired()
{
    ASSERT(!m_suspended);
    ASSERT(m_parser);

    // FIXME: The timer class should handle timer priorities instead of this code.
    // If a layout is scheduled, wait again to let the layout timer run first.
    if (m_parser->document()->isLayoutPending()) {
        m_continueNextChunkTimer.startOneShot(0_s);
        return;
    }
    m_parser->resumeParsingAfterYield();
}

static bool parsingProgressedSinceLastYield(PumpSession& session)
{
    // Only yield if there has been progress since last yield.
    if (session.processedTokens > session.processedTokensOnLastYieldBeforeScript) {
        session.processedTokensOnLastYieldBeforeScript = session.processedTokens;
        return true;
    }
    return false;
}

bool HTMLParserScheduler::shouldYieldBeforeExecutingScript(const ScriptElement* scriptElement, PumpSession& session)
{
    // If we've never painted before and a layout is pending, yield prior to running
    // scripts to give the page a chance to paint earlier.
    RefPtr<Document> document = m_parser->document();

    session.didSeeScript = true;

    if (!document->body())
        return false;

    if (!document->frame() || !document->frame()->script().canExecuteScripts(ReasonForCallingCanExecuteScripts::NotAboutToExecuteScript))
        return false;

    if (!document->haveStylesheetsLoaded())
        return false;

    if (UNLIKELY(m_documentHasActiveParserYieldTokens))
        return true;

    // Yield if we have never painted and there is meaningful content
    if (document->view() && !document->view()->hasEverPainted() && document->view()->isVisuallyNonEmpty())
        return parsingProgressedSinceLastYield(session);

    auto elapsedTime = MonotonicTime::now() - session.startTime;

    constexpr auto elapsedTimeLimit = 16_ms;
    // Require at least some new parsed content before yielding.
    constexpr auto tokenLimit = 256;
    // Don't yield on very short inline scripts. This is an imperfect way to try to guess the execution cost.
    constexpr auto inlineScriptLengthLimit = 1024;

    if (elapsedTime < elapsedTimeLimit)
        return false;
    if (session.processedTokens < tokenLimit)
        return false;

    if (scriptElement) {
        // Async and deferred scripts are not executed by the parser.
        if (scriptElement->hasAsyncAttribute() || scriptElement->hasDeferAttribute())
            return false;
        if (!scriptElement->hasSourceAttribute() && scriptElement->scriptContent().length() < inlineScriptLengthLimit)
            return false;
    }

    return true;
}

void HTMLParserScheduler::scheduleForResume()
{
    ASSERT(!m_suspended);
    m_continueNextChunkTimer.startOneShot(0_s);
}

void HTMLParserScheduler::suspend()
{
    ASSERT(!m_suspended);
    ASSERT(!m_isSuspendedWithActiveTimer);
#if ASSERT_ENABLED
    m_suspended = true;
#endif

    if (!m_continueNextChunkTimer.isActive())
        return;
    m_isSuspendedWithActiveTimer = true;
    m_continueNextChunkTimer.stop();
}

void HTMLParserScheduler::resume()
{
    ASSERT(m_suspended);
    ASSERT(!m_continueNextChunkTimer.isActive());
#if ASSERT_ENABLED
    m_suspended = false;
#endif

    if (!m_isSuspendedWithActiveTimer)
        return;
    m_isSuspendedWithActiveTimer = false;
    m_continueNextChunkTimer.startOneShot(0_s);
}

}
