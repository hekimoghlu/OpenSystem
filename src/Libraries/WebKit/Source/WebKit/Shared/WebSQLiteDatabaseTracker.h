/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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

#include <WebCore/SQLiteDatabaseTrackerClient.h>
#include <wtf/Forward.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>
#include <wtf/RunLoop.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

// Use eager initialization for the WeakPtrFactory since we construct WeakPtrs from a non-main thread.
class WebSQLiteDatabaseTracker final : public WebCore::SQLiteDatabaseTrackerClient, public RefCounted<WebSQLiteDatabaseTracker>, public CanMakeWeakPtr<WebSQLiteDatabaseTracker, WeakPtrFactoryInitialization::Eager> {
    WTF_MAKE_NONCOPYABLE(WebSQLiteDatabaseTracker)
public:
    // IsHoldingLockedFilesHandler may get called on a non-main thread, but while holding a Lock.
    using IsHoldingLockedFilesHandler = Function<void(bool)>;
    static Ref<WebSQLiteDatabaseTracker> create(IsHoldingLockedFilesHandler&&);

    ~WebSQLiteDatabaseTracker();

    void setIsSuspended(bool);

private:
    explicit WebSQLiteDatabaseTracker(IsHoldingLockedFilesHandler&&);

    void setIsHoldingLockedFiles(bool) WTF_REQUIRES_LOCK(m_lock);

    // WebCore::SQLiteDatabaseTrackerClient.
    void willBeginFirstTransaction() final;
    void didFinishLastTransaction() final;

    IsHoldingLockedFilesHandler m_isHoldingLockedFilesHandler;
    Lock m_lock;
    uint64_t m_currentHystererisID WTF_GUARDED_BY_LOCK(m_lock) { 0 };
    bool m_isSuspended WTF_GUARDED_BY_LOCK(m_lock) { false };
};

} // namespace WebKit
