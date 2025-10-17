/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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

#include "NestingLevelIncrementer.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

#if PLATFORM(IOS_FAMILY)
#include "WebCoreThread.h"
#endif

namespace WebCore {

class Document;
class HTMLDocumentParser;
class ScriptElement;

class ActiveParserSession {
public:
    explicit ActiveParserSession(Document*);
    ~ActiveParserSession();

private:
    RefPtr<Document> m_document;
};

class PumpSession : public NestingLevelIncrementer, public ActiveParserSession {
public:
    PumpSession(unsigned& nestingLevel, Document*);
    ~PumpSession();

    unsigned processedTokens { 0 };
    unsigned processedTokensOnLastCheck { 0 };
    unsigned processedTokensOnLastYieldBeforeScript { 0 };
    MonotonicTime startTime { MonotonicTime::now() };
    bool didSeeScript { false };
};

class HTMLParserScheduler final : public RefCounted<HTMLParserScheduler> {
    WTF_MAKE_TZONE_ALLOCATED(HTMLParserScheduler);
    WTF_MAKE_NONCOPYABLE(HTMLParserScheduler);
public:
    static Ref<HTMLParserScheduler> create(HTMLDocumentParser&);
    ~HTMLParserScheduler();

    void detach();

    bool shouldYieldBeforeToken(PumpSession& session)
    {
#if PLATFORM(IOS_FAMILY)
        if (WebThreadShouldYield())
            return true;
#endif
        if (UNLIKELY(m_documentHasActiveParserYieldTokens))
            return true;

        if (UNLIKELY(session.processedTokens > session.processedTokensOnLastCheck + numberOfTokensBeforeCheckingForYield || session.didSeeScript))
            return checkForYield(session);

        ++session.processedTokens;
        return false;
    }
    bool shouldYieldBeforeExecutingScript(const ScriptElement*, PumpSession&);

    void scheduleForResume();
    bool isScheduledForResume() const { return m_isSuspendedWithActiveTimer || m_continueNextChunkTimer.isActive() || m_documentHasActiveParserYieldTokens; }

    void suspend();
    void resume();

    void didBeginYieldingParser()
    {
        ASSERT(!m_documentHasActiveParserYieldTokens);
        m_documentHasActiveParserYieldTokens = true;
    }

    void didEndYieldingParser()
    {
        ASSERT(m_documentHasActiveParserYieldTokens);
        m_documentHasActiveParserYieldTokens = false;

        if (!isScheduledForResume())
            scheduleForResume();
    }

private:
    explicit HTMLParserScheduler(HTMLDocumentParser&);

    static const unsigned numberOfTokensBeforeCheckingForYield = 4096; // Performance optimization

    void continueNextChunkTimerFired();

    bool checkForYield(PumpSession& session)
    {
        session.processedTokensOnLastCheck = session.processedTokens;
        session.didSeeScript = false;

        Seconds elapsedTime = MonotonicTime::now() - session.startTime;
        return elapsedTime > m_parserTimeLimit;
    }

    CheckedPtr<HTMLDocumentParser> m_parser;

    Seconds m_parserTimeLimit;
    Timer m_continueNextChunkTimer;
    bool m_isSuspendedWithActiveTimer;
#if ASSERT_ENABLED
    bool m_suspended;
#endif
    bool m_documentHasActiveParserYieldTokens { false };
};

} // namespace WebCore
