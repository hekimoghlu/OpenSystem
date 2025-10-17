/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

#include "PageIdentifier.h"
#include "VisibilityChangeClient.h"
#include "WakeLockType.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class SleepDisabler;
class WakeLockSentinel;

class WakeLockManager final : public VisibilityChangeClient {
    WTF_MAKE_TZONE_ALLOCATED(WakeLockManager);
public:
    explicit WakeLockManager(Document&);
    ~WakeLockManager();

    void ref() const final;
    void deref() const final;

    void addWakeLock(Ref<WakeLockSentinel>&&, std::optional<PageIdentifier>);
    void removeWakeLock(WakeLockSentinel&);

    void releaseAllLocks(WakeLockType);

private:
    void visibilityStateChanged() final;

    Document& m_document;
    HashMap<WakeLockType, Vector<RefPtr<WakeLockSentinel>>, WTF::IntHash<WakeLockType>, WTF::StrongEnumHashTraits<WakeLockType>> m_wakeLocks;
    std::unique_ptr<SleepDisabler> m_screenLockDisabler;
};

} // namespace WebCore
