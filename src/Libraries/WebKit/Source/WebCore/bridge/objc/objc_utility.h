/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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

#include <CoreFoundation/CoreFoundation.h>

#include "objc_header.h"
#include <JavaScriptCore/Error.h>
#include <JavaScriptCore/JSObject.h>

OBJC_CLASS NSString;

namespace JSC {
namespace Bindings {

typedef union {
    CFTypeRef objectValue;
    bool booleanValue;
    char charValue;
    short shortValue;
    int intValue;
    long longValue;
    long long longLongValue;
    float floatValue;
    double doubleValue;
} ObjcValue;

typedef enum {
    ObjcVoidType,
    ObjcObjectType,
    ObjcCharType,
    ObjcUnsignedCharType,
    ObjcShortType,
    ObjcUnsignedShortType,
    ObjcIntType,
    ObjcUnsignedIntType,
    ObjcLongType,
    ObjcUnsignedLongType,
    ObjcLongLongType,
    ObjcUnsignedLongLongType,
    ObjcFloatType,
    ObjcDoubleType,
    ObjcBoolType,
    ObjcInvalidType
} ObjcValueType;

class RootObject;

ObjcValue convertValueToObjcValue(JSGlobalObject*, JSValue, ObjcValueType);
JSValue convertNSStringToString(JSGlobalObject* lexicalGlobalObject, NSString *nsstring);
JSValue convertObjcValueToValue(JSGlobalObject*, void* buffer, ObjcValueType, RootObject*);
ObjcValueType objcValueTypeForType(const char *type);

Exception *throwError(JSGlobalObject*, ThrowScope&, NSString *message);

} // namespace Bindings
} // namespace JSC
