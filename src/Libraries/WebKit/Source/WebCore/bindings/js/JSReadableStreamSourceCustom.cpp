/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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
#include "JSReadableStreamSource.h"

#include "JSDOMPromiseDeferred.h"

namespace WebCore {
using namespace JSC;

JSValue JSReadableStreamSource::start(JSGlobalObject& lexicalGlobalObject, CallFrame& callFrame, Ref<DeferredPromise>&& promise)
{
    VM& vm = lexicalGlobalObject.vm();
    
    // FIXME: Why is it ok to ASSERT the argument count here?
    ASSERT(callFrame.argumentCount());
    JSReadableStreamDefaultController* controller = jsDynamicCast<JSReadableStreamDefaultController*>(callFrame.uncheckedArgument(0));
    ASSERT(controller);

    m_controller.set(vm, this, controller);

    wrapped().start(ReadableStreamDefaultController(controller), WTFMove(promise));

    return jsUndefined();
}

JSValue JSReadableStreamSource::pull(JSGlobalObject&, CallFrame&, Ref<DeferredPromise>&& promise)
{
    wrapped().pull(WTFMove(promise));
    return jsUndefined();
}

JSValue JSReadableStreamSource::controller(JSGlobalObject&) const
{
    ASSERT_NOT_REACHED();
    return jsUndefined();
}

}
