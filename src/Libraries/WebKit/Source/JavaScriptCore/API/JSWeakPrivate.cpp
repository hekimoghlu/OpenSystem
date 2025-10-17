/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include "JSWeakPrivate.h"

#include "APICast.h"
#include "IntegrityInlines.h"
#include "Weak.h"
#include <wtf/ThreadSafeRefCounted.h>

using namespace JSC;

struct OpaqueJSWeak : ThreadSafeRefCounted<OpaqueJSWeak> {
    OpaqueJSWeak(JSObject* object)
        : weak(object)
    {
    }
    
    Weak<JSObject> weak;
};

JSWeakRef JSWeakCreate(JSContextGroupRef contextGroup, JSObjectRef objectRef)
{
    VM* vm = toJS(contextGroup);
    JSLockHolder locker(vm);
    return new OpaqueJSWeak(toJS(objectRef));
}

void JSWeakRetain(JSContextGroupRef contextGroup, JSWeakRef weakRef)
{
    VM* vm = toJS(contextGroup);
    JSLockHolder locker(vm);
    weakRef->ref();
}

void JSWeakRelease(JSContextGroupRef contextGroup, JSWeakRef weakRef)
{
    VM* vm = toJS(contextGroup);
    JSLockHolder locker(vm);
    weakRef->deref();
}

JSObjectRef JSWeakGetObject(JSWeakRef weakRef)
{
    return toRef(weakRef->weak.get());
}

