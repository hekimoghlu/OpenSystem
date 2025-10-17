/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
#ifndef JSWeakObjectMapRefInternal_h
#define JSWeakObjectMapRefInternal_h

#include "WeakGCMap.h"
#include <wtf/RefCounted.h>

namespace JSC {

class JSObject;

}

typedef void (*JSWeakMapDestroyedCallback)(struct OpaqueJSWeakObjectMap*, void*);

typedef JSC::WeakGCMap<void*, JSC::JSObject> WeakMapType;

#define OPAQUE_JSWEAK_OBJECT_MAP_METHOD(method) \
    WTF_VTBL_FUNCPTR_PTRAUTH_STR("OpaqueJSWeakObjectMap." #method) method

struct OpaqueJSWeakObjectMap : public RefCounted<OpaqueJSWeakObjectMap> {
public:
    static Ref<OpaqueJSWeakObjectMap> create(JSC::VM& vm, void* data, JSWeakMapDestroyedCallback callback)
    {
        return adoptRef(*new OpaqueJSWeakObjectMap(vm, data, callback));
    }

    WeakMapType& map() { return m_map; }
    
    ~OpaqueJSWeakObjectMap()
    {
        m_callback(this, m_data);
    }

private:
    OpaqueJSWeakObjectMap(JSC::VM& vm, void* data, JSWeakMapDestroyedCallback callback)
        : m_map(vm)
        , m_data(data)
        , m_callback(callback)
    {
    }
    WeakMapType m_map;
    void* m_data;
    JSWeakMapDestroyedCallback OPAQUE_JSWEAK_OBJECT_MAP_METHOD(m_callback);
};

#undef OPAQUE_JSWEAK_OBJECT_MAP_METHOD

#endif // JSWeakObjectMapInternal_h
