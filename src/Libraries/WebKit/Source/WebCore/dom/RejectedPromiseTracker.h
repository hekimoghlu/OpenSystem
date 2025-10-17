/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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

#include <JavaScriptCore/JSPromise.h>
#include <JavaScriptCore/WeakGCMap.h>
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>

namespace JSC {
class VM;
}

namespace WebCore {

class DOMPromise;
class JSDOMGlobalObject;
class ScriptExecutionContext;
class UnhandledPromise;

class RejectedPromiseTracker final : public CanMakeCheckedPtr<RejectedPromiseTracker> {
    WTF_MAKE_TZONE_ALLOCATED(RejectedPromiseTracker);
    WTF_MAKE_NONCOPYABLE(RejectedPromiseTracker);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RejectedPromiseTracker);
public:
    explicit RejectedPromiseTracker(ScriptExecutionContext&, JSC::VM&);
    ~RejectedPromiseTracker();

    void promiseRejected(JSDOMGlobalObject&, JSC::JSPromise&);
    void promiseHandled(JSDOMGlobalObject&, JSC::JSPromise&);

    void processQueueSoon();

private:
    void reportUnhandledRejections(Vector<UnhandledPromise>&&);
    void reportRejectionHandled(Ref<DOMPromise>&&);

    WeakRef<ScriptExecutionContext> m_context;
    Vector<UnhandledPromise> m_aboutToBeNotifiedRejectedPromises;
    JSC::WeakGCMap<JSC::JSPromise*, JSC::JSPromise> m_outstandingRejectedPromises;
};

} // namespace WebCore
