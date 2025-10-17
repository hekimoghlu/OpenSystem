/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
#include "JSWebExtensionWrapper.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#include "JSWebExtensionWrappable.h"
#include "WebFrame.h"
#include "WebPage.h"
#include <JavaScriptCore/JSObjectRef.h>
#include <JavaScriptCore/JSWeakObjectMapRefPrivate.h>

namespace WebKit {

static HashMap<JSGlobalContextRef, JSWeakObjectMapRef>& wrapperCache()
{
    static NeverDestroyed<HashMap<JSGlobalContextRef, JSWeakObjectMapRef>> wrappers;
    return wrappers;
}

static void cacheMapDestroyed(JSWeakObjectMapRef map, void* context)
{
    wrapperCache().remove(static_cast<JSGlobalContextRef>(context));
}

static inline JSWeakObjectMapRef wrapperCacheMap(JSContextRef context)
{
    auto globalContext = JSContextGetGlobalContext(context);
    return wrapperCache().ensure(globalContext, [&] {
        return JSWeakObjectMapCreate(globalContext, globalContext, cacheMapDestroyed);
    }).iterator->value;
}

static inline JSValueRef getCachedWrapper(JSContextRef context, JSWeakObjectMapRef wrappers, JSWebExtensionWrappable* object)
{
    ASSERT(context);
    ASSERT(wrappers);
    ASSERT(object);

    if (auto wrapper = JSWeakObjectMapGet(context, wrappers, object)) {
        // Check if the wrapper is still valid. Objects invalidated through finalize
        // will not get removed from the map automatically.
        if (JSObjectGetPrivate(wrapper))
            return wrapper;

        // Remove from the map, since the object is invalid.
        JSWeakObjectMapRemove(context, wrappers, object);
    }

    return nullptr;
}

JSValueRef JSWebExtensionWrapper::wrap(JSContextRef context, JSWebExtensionWrappable* object)
{
    ASSERT(context);

    if (!object)
        return JSValueMakeNull(context);

    auto wrappers = wrapperCacheMap(context);
    if (auto result = getCachedWrapper(context, wrappers, object))
        return result;

    auto objectClass = object->wrapperClass();
    ASSERT(objectClass);

    auto wrapper = JSObjectMake(context, objectClass, object);
    ASSERT(wrapper);

    JSWeakObjectMapSet(context, wrappers, object, wrapper);

    return wrapper;
}

JSWebExtensionWrappable* JSWebExtensionWrapper::unwrap(JSContextRef context, JSValueRef value)
{
    ASSERT(context);
    ASSERT(value);

    if (!context || !value)
        return nullptr;

    return static_cast<JSWebExtensionWrappable*>(JSObjectGetPrivate(JSValueToObject(context, value, nullptr)));
}

static JSWebExtensionWrappable* unwrapObject(JSObjectRef object)
{
    ASSERT(object);

    auto* wrappable = static_cast<JSWebExtensionWrappable*>(JSObjectGetPrivate(object));
    ASSERT(wrappable);
    return wrappable;
}

void JSWebExtensionWrapper::initialize(JSContextRef, JSObjectRef object)
{
    if (auto* wrappable = unwrapObject(object))
        wrappable->ref();
}

void JSWebExtensionWrapper::finalize(JSObjectRef object)
{
    if (auto* wrappable = unwrapObject(object)) {
        JSObjectSetPrivate(object, nullptr);
        wrappable->deref();
    }
}

RefPtr<WebFrame> toWebFrame(JSContextRef context)
{
    ASSERT(context);
    return WebFrame::frameForContext(JSContextGetGlobalContext(context));
}

RefPtr<WebPage> toWebPage(JSContextRef context)
{
    ASSERT(context);
    auto frame = toWebFrame(context);
    return frame ? frame->page() : nullptr;
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
