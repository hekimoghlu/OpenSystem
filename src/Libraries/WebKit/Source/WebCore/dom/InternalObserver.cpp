/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
#include "InternalObserver.h"

#include "ContextDestructionObserverInlines.h"
#include "JSDOMExceptionHandling.h"
#include <JavaScriptCore/Exception.h>
#include <JavaScriptCore/JSCJSValue.h>
#include <JavaScriptCore/JSGlobalObject.h>

namespace WebCore {

void InternalObserver::error(JSC::JSValue value)
{
    RefPtr context = scriptExecutionContext();
    if (!context)
        return;

    auto* globalObject = context->globalObject();
    if (!globalObject)
        return;

    Ref vm = globalObject->vm();
    JSC::JSLockHolder lock(vm);

    reportException(globalObject, JSC::Exception::create(vm, value));
}

} // namespace WebCore
