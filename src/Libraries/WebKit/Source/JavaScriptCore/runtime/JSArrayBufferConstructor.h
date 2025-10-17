/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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

#include "ArrayBuffer.h"
#include "InternalFunction.h"

namespace JSC {

class JSArrayBufferPrototype;
class GetterSetter;

template<ArrayBufferSharingMode sharingMode>
class JSGenericArrayBufferConstructor final : public InternalFunction {
public:
    using Base = InternalFunction;

    static JSGenericArrayBufferConstructor* create(VM& vm, Structure* structure, JSArrayBufferPrototype* prototype)
    {
        JSGenericArrayBufferConstructor* result =
            new (NotNull, allocateCell<JSGenericArrayBufferConstructor>(vm)) JSGenericArrayBufferConstructor(vm, structure);
        result->finishCreation(vm, prototype);
        return result;
    }

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype);
    
    static const ClassInfo s_info; // This is never accessed directly, since that would break linkage on some compilers.
    static const ClassInfo* info();

    static EncodedJSValue constructImpl(JSGlobalObject*, CallFrame*);

private:
    JSGenericArrayBufferConstructor(VM&, Structure*);
    void finishCreation(VM&, JSArrayBufferPrototype*);
};

using JSArrayBufferConstructor = JSGenericArrayBufferConstructor<ArrayBufferSharingMode::Default>;
using JSSharedArrayBufferConstructor = JSGenericArrayBufferConstructor<ArrayBufferSharingMode::Shared>;
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSArrayBufferConstructor, InternalFunction);
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSSharedArrayBufferConstructor, InternalFunction);

} // namespace JSC
