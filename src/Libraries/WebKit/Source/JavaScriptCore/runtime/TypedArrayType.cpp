/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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
#include "TypedArrayType.h"

#include "JSTypedArrayConstructors.h"

namespace JSC {

const uint8_t logElementSizes[] = {
#define JSC_ELEMENT_SIZE(type) logElementSize(Type ## type),
FOR_EACH_TYPED_ARRAY_TYPE(JSC_ELEMENT_SIZE)
#undef JSC_ELEMENT_SIZE
};

const ClassInfo* constructorClassInfoForType(TypedArrayType type)
{
    switch (type) {
    case NotTypedArray:
        return nullptr;
    case TypeInt8:
        return JSInt8ArrayConstructor::info();
    case TypeUint8:
        return JSUint8ArrayConstructor::info();
    case TypeUint8Clamped:
        return JSUint8ClampedArrayConstructor::info();
    case TypeInt16:
        return JSInt16ArrayConstructor::info();
    case TypeUint16:
        return JSUint16ArrayConstructor::info();
    case TypeInt32:
        return JSInt32ArrayConstructor::info();
    case TypeUint32:
        return JSUint32ArrayConstructor::info();
    case TypeFloat16:
        return JSFloat16ArrayConstructor::info();
    case TypeFloat32:
        return JSFloat32ArrayConstructor::info();
    case TypeFloat64:
        return JSFloat64ArrayConstructor::info();
    case TypeBigInt64:
        return JSBigInt64ArrayConstructor::info();
    case TypeBigUint64:
        return JSBigUint64ArrayConstructor::info();
    case TypeDataView:
        return JSDataViewConstructor::info();
    }
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

} // namespace JSC

namespace WTF {

using namespace JSC;

void printInternal(PrintStream& out, TypedArrayType type)
{
    switch (type) {
    case NotTypedArray:
        out.print("NotTypedArray");
        return;
    case TypeInt8:
        out.print("TypeInt8");
        return;
    case TypeInt16:
        out.print("TypeInt16");
        return;
    case TypeInt32:
        out.print("TypeInt32");
        return;
    case TypeUint8:
        out.print("TypeUint8");
        return;
    case TypeUint8Clamped:
        out.print("TypeUint8Clamped");
        return;
    case TypeUint16:
        out.print("TypeUint16");
        return;
    case TypeUint32:
        out.print("TypeUint32");
        return;
    case TypeFloat16:
        out.print("TypeFloat16");
        return;
    case TypeFloat32:
        out.print("TypeFloat32");
        return;
    case TypeFloat64:
        out.print("TypeFloat64");
        return;
    case TypeBigInt64:
        out.print("TypeBigInt64");
        return;
    case TypeBigUint64:
        out.print("TypeBigUint64");
        return;
    case TypeDataView:
        out.print("TypeDataView");
        return;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

