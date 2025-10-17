/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
#include "JSWeakObjectMapRefPrivate.h"

#include "APICast.h"
#include "JSCallbackObject.h"
#include "JSWeakObjectMapRefInternal.h"
#include "WeakGCMapInlines.h"

using namespace JSC;

#ifdef __cplusplus
extern "C" {
#endif

JSWeakObjectMapRef JSWeakObjectMapCreate(JSContextRef context, void* privateData, JSWeakMapDestroyedCallback callback)
{
    JSGlobalObject* globalObject = toJS(context);
    VM& vm = globalObject->vm();
    JSLockHolder locker(vm);
    auto map = OpaqueJSWeakObjectMap::create(vm, privateData, callback);
    globalObject->registerWeakMap(map.ptr());
    return map.ptr();
}

void JSWeakObjectMapSet(JSContextRef ctx, JSWeakObjectMapRef map, void* key, JSObjectRef object)
{
    if (!ctx) {
        ASSERT_NOT_REACHED();
        return;
    }
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder locker(vm);
    JSObject* obj = toJS(object);
    if (!obj)
        return;
    ASSERT(obj->inherits<JSGlobalProxy>()
        || obj->inherits<JSCallbackObject<JSGlobalObject>>()
        || obj->inherits<JSCallbackObject<JSNonFinalObject>>());
    map->map().set(key, obj);
}

JSObjectRef JSWeakObjectMapGet(JSContextRef ctx, JSWeakObjectMapRef map, void* key)
{
    if (!ctx) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }
    JSGlobalObject* globalObject = toJS(ctx);
    JSLockHolder locker(globalObject);
    return toRef(jsCast<JSObject*>(map->map().get(key)));
}

void JSWeakObjectMapRemove(JSContextRef ctx, JSWeakObjectMapRef map, void* key)
{
    if (!ctx) {
        ASSERT_NOT_REACHED();
        return;
    }
    JSGlobalObject* globalObject = toJS(ctx);
    JSLockHolder locker(globalObject);
    map->map().remove(key);
}

#ifdef __cplusplus
}
#endif
