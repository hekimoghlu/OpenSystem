/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/URL.h>
#include <wtf/WeakHashSet.h>

namespace WebKit {

enum class IsMainFrame : bool;

class FrameLoadStateObserver : public AbstractRefCountedAndCanMakeWeakPtr<FrameLoadStateObserver> {
public:
    virtual ~FrameLoadStateObserver() = default;

    virtual void didReceiveProvisionalURL(const URL&) { }
    virtual void didStartProvisionalLoad(const URL&) { }
    virtual void didFailProvisionalLoad(const URL&) { }
    virtual void didFailLoad(const URL&) { }
    virtual void didCancelProvisionalLoad() { }
    virtual void didCommitProvisionalLoad() { }
    virtual void didCommitProvisionalLoad(IsMainFrame) { }
    virtual void didFinishLoad(IsMainFrame, const URL&) { }
};

class FrameLoadState {
public:
    FrameLoadState(IsMainFrame isMainFrame)
        : m_isMainFrame(isMainFrame) { }

    ~FrameLoadState();

    enum class State {
        Provisional,
        Committed,
        Finished
    };

    void addObserver(FrameLoadStateObserver&);
    void removeObserver(FrameLoadStateObserver&);

    void didStartProvisionalLoad(const URL&);
    void didExplicitOpen(const URL&);
    void didReceiveServerRedirectForProvisionalLoad(const URL&);
    void didFailProvisionalLoad();
    void didSuspend();

    void didCommitLoad();
    void didFinishLoad();
    void didFailLoad();

    void didSameDocumentNotification(const URL&);

    State state() const { return m_state; }
    const URL& url() const { return m_url; }
    void setURL(const URL&);
    const URL& provisionalURL() const { return m_provisionalURL; }

    void setUnreachableURL(const URL&);
    const URL& unreachableURL() const { return m_unreachableURL; }

    IsMainFrame isMainFrame() const { return m_isMainFrame; }

private:
    void forEachObserver(const Function<void(FrameLoadStateObserver&)>&);

    const IsMainFrame m_isMainFrame;
    State m_state { State::Finished };
    URL m_url;
    URL m_provisionalURL;
    URL m_unreachableURL;
    URL m_lastUnreachableURL;
    WeakHashSet<FrameLoadStateObserver> m_observers;
};

} // namespace WebKit
