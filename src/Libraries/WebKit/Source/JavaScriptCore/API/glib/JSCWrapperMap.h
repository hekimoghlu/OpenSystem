/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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

#include "JSBase.h"
#include "VM.h"
#include "WeakGCMap.h"
#include <glib.h>
#include <wtf/HashMap.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/CString.h>

typedef struct _JSCClass JSCClass;
typedef struct _JSCContext JSCContext;
typedef struct _JSCValue JSCValue;

namespace JSC {

class JSObject;

class WrapperMap {
    WTF_MAKE_FAST_ALLOCATED;
public:
    explicit WrapperMap(JSGlobalContextRef);
    ~WrapperMap();

    GRefPtr<JSCValue> gobjectWrapper(JSCContext*, JSValueRef);
    void unwrap(JSValueRef);

    void registerClass(JSCClass*);
    JSCClass* registeredClass(JSClassRef) const;

    JSObject* createJSWrapper(JSGlobalContextRef, JSClassRef, JSValueRef prototype, gpointer, GDestroyNotify);
    JSGlobalContextRef createContextWithJSWrapper(JSContextGroupRef, JSClassRef, JSValueRef prototype, gpointer, GDestroyNotify);
    JSObject* jsWrapper(gpointer wrappedObject) const;
    gpointer wrappedObject(JSGlobalContextRef, JSObjectRef) const;

private:
    UncheckedKeyHashMap<JSValueRef, JSCValue*> m_cachedGObjectWrappers;
    std::unique_ptr<JSC::WeakGCMap<gpointer, JSC::JSObject>> m_cachedJSWrappers;
    UncheckedKeyHashMap<JSClassRef, GRefPtr<JSCClass>> m_classMap;
};

} // namespace JSC
