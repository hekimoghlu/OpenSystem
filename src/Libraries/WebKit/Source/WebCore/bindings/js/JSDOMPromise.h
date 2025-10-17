/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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

#include "JSDOMGuardedObject.h"
#include <JavaScriptCore/JSPromise.h>

namespace WebCore {

class DOMPromise : public DOMGuarded<JSC::JSPromise> {
public:
    static Ref<DOMPromise> create(JSDOMGlobalObject& globalObject, JSC::JSPromise& promise)
    {
        return adoptRef(*new DOMPromise(globalObject, promise));
    }

    JSC::JSPromise* promise() const
    {
        ASSERT(!isSuspended());
        return guarded();
    }

    enum class IsCallbackRegistered : bool { No, Yes };
    IsCallbackRegistered whenSettled(std::function<void()>&&);
    JSC::JSValue result() const;

    enum class Status { Pending, Fulfilled, Rejected };
    Status status() const;

    static IsCallbackRegistered whenPromiseIsSettled(JSDOMGlobalObject*, JSC::JSObject* promise, Function<void()>&&);

private:
    DOMPromise(JSDOMGlobalObject& globalObject, JSC::JSPromise& promise)
        : DOMGuarded<JSC::JSPromise>(globalObject, promise)
    {
    }
};

} // namespace WebCore
