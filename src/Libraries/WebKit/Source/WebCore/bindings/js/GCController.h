/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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

#include "Timer.h"
#include <JavaScriptCore/DeleteAllCodeEffort.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {
class VM;
}

namespace WebCore {

class GCController {
    WTF_MAKE_TZONE_ALLOCATED(GCController);
    WTF_MAKE_NONCOPYABLE(GCController);
    friend class WTF::NeverDestroyed<GCController>;
public:
    WEBCORE_EXPORT static GCController& singleton();
    WEBCORE_EXPORT static void dumpHeapForVM(JSC::VM&);

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    WEBCORE_EXPORT void garbageCollectSoon();
    WEBCORE_EXPORT void garbageCollectNow(); // It's better to call garbageCollectSoon, unless you have a specific reason not to.
    WEBCORE_EXPORT void garbageCollectNowIfNotDoneRecently();
    void garbageCollectOnNextRunLoop();

    WEBCORE_EXPORT void garbageCollectOnAlternateThreadForDebugging(bool waitUntilDone); // Used for stress testing.
    WEBCORE_EXPORT void setJavaScriptGarbageCollectorTimerEnabled(bool);
    WEBCORE_EXPORT void deleteAllCode(JSC::DeleteAllCodeEffort);
    WEBCORE_EXPORT void deleteAllLinkedCode(JSC::DeleteAllCodeEffort);

    WEBCORE_EXPORT void dumpHeap();

private:
    GCController(); // Use singleton() instead.

    void gcTimerFired();
    Timer m_GCTimer;
};

} // namespace WebCore
