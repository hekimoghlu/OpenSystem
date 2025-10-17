/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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

#include "PendingScriptClient.h"
#include "Timer.h"
#include <wtf/CheckedRef.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class Document;
class ScriptElement;
class LoadableScript;
class WeakPtrImplWithEventTargetData;

class ScriptRunner final : public PendingScriptClient, public CanMakeCheckedPtr<ScriptRunner> {
    WTF_MAKE_TZONE_ALLOCATED(ScriptRunner);
    WTF_MAKE_NONCOPYABLE(ScriptRunner);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ScriptRunner);
public:
    explicit ScriptRunner(Document&);
    ~ScriptRunner();

    void ref() const;
    void deref() const;

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    enum ExecutionType { ASYNC_EXECUTION, IN_ORDER_EXECUTION };
    void queueScriptForExecution(ScriptElement&, LoadableScript&, ExecutionType);
    bool hasPendingScripts() const { return !m_scriptsToExecuteSoon.isEmpty() || !m_scriptsToExecuteInOrder.isEmpty() || !m_pendingAsyncScripts.isEmpty(); }
    void suspend();
    void resume();
    void notifyScriptReady(ScriptElement*, ExecutionType);

    void didBeginYieldingParser() { suspend(); }
    void didEndYieldingParser() { resume(); }

    void documentFinishedParsing();

    void clearPendingScripts();

private:
    void timerFired();

    void notifyFinished(PendingScript&) override;

    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
    Vector<Ref<PendingScript>> m_scriptsToExecuteInOrder;
    Vector<RefPtr<PendingScript>> m_scriptsToExecuteSoon; // http://www.whatwg.org/specs/web-apps/current-work/#set-of-scripts-that-will-execute-as-soon-as-possible
    UncheckedKeyHashSet<Ref<PendingScript>> m_pendingAsyncScripts;
    Timer m_timer;
};

}
