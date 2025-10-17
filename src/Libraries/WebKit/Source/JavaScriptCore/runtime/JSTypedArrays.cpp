/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
#include "JSTypedArrays.h"

#include "GenericTypedArrayViewInlines.h"
#include "JSCInlines.h"
#include "JSGenericTypedArrayViewInlines.h"
#include "JSGenericTypedArrayViewConstructorInlines.h"

namespace JSC {

#define MAKE_CONSTRUCTORS(Class) \
    JSC_DEFINE_HOST_FUNCTION(call##Class, (JSGlobalObject* globalObject, CallFrame* callFrame)) { \
        return callGenericTypedArrayViewImpl<JS##Class>(globalObject, callFrame); \
    } \
    JSC_DEFINE_HOST_FUNCTION(construct##Class, (JSGlobalObject* globalObject, CallFrame* callFrame)) { \
        return constructGenericTypedArrayViewImpl<JS##Class>(globalObject, callFrame); \
    }

#undef MAKE_S_INFO
#define MAKE_S_INFO(type) \
    template<> const ClassInfo JS##type##Array::s_info = { \
        #type "Array"_s, &JS##type##Array::Base::s_info, nullptr, nullptr, \
        CREATE_METHOD_TABLE(JS##type##Array) \
    }; \
    const ClassInfo* get##type##ArrayClassInfo() { return &JS##type##Array::s_info; } \
    template<> const ClassInfo JSResizableOrGrowableShared##type##Array::s_info = { \
        #type "Array"_s, &JSResizableOrGrowableShared##type##Array::Base::s_info, nullptr, nullptr, \
        CREATE_METHOD_TABLE(JSResizableOrGrowableShared##type##Array) \
    }; \
    const ClassInfo* getResizableOrGrowableShared##type##ArrayClassInfo() { return &JSResizableOrGrowableShared##type##Array::s_info; } \
    MAKE_CONSTRUCTORS(type##Array)

MAKE_S_INFO(Int8);
MAKE_S_INFO(Int16);
MAKE_S_INFO(Int32);
MAKE_S_INFO(Uint8);
MAKE_S_INFO(Uint8Clamped);
MAKE_S_INFO(Uint16);
MAKE_S_INFO(Uint32);
MAKE_S_INFO(Float16);
MAKE_S_INFO(Float32);
MAKE_S_INFO(Float64);
MAKE_S_INFO(BigInt64);
MAKE_S_INFO(BigUint64);

MAKE_CONSTRUCTORS(DataView);

} // namespace JSC

