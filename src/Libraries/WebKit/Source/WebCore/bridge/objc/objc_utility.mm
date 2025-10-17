/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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
#import "config.h"
#import "objc_utility.h"

#import "WebScriptObjectProtocol.h"
#import "objc_instance.h"
#import "runtime_array.h"
#import "runtime_object.h"
#import <JavaScriptCore/JSGlobalObjectInlines.h>
#import <JavaScriptCore/JSLock.h>
#import <wtf/Assertions.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

#if !defined(_C_LNG_LNG)
#define _C_LNG_LNG 'q'
#endif

#if !defined(_C_ULNG_LNG)
#define _C_ULNG_LNG 'Q'
#endif

#if !defined(_C_CONST)
#define _C_CONST 'r'
#endif

#if !defined(_C_BYCOPY)
#define _C_BYCOPY 'O'
#endif

#if !defined(_C_BYREF)
#define _C_BYREF 'R'
#endif

#if !defined(_C_ONEWAY)
#define _C_ONEWAY 'V'
#endif

#if !defined(_C_GCINVISIBLE)
#define _C_GCINVISIBLE '!'
#endif

namespace JSC {
namespace Bindings {

/*

    JavaScript to   ObjC
    Number          coerced to char, short, int, long, float, double, or NSNumber, as appropriate
    String          NSString
    wrapper         id
    Object          WebScriptObject
    null            NSNull
    [], other       exception

*/

static long long convertDoubleToLongLong(double d)
{
    // Ensure that non-x86_64 ports will match x86_64 semantics which allows for overflow.
    long long potentiallyOverflowed = (unsigned long long)d;
    if (potentiallyOverflowed < (long long)d)
        return potentiallyOverflowed;
    return (long long)d;
}

ObjcValue convertValueToObjcValue(JSGlobalObject* lexicalGlobalObject, JSValue value, ObjcValueType type)
{
    ObjcValue result;
    double d = 0;

    if (value.isNumber() || value.isString() || value.isBoolean()) {
        JSLockHolder lock(lexicalGlobalObject);
        d = value.toNumber(lexicalGlobalObject);
    }

    switch (type) {
        case ObjcObjectType: {
            VM& vm = lexicalGlobalObject->vm();
            JSLockHolder lock(vm);
            
            JSGlobalObject *originGlobalObject = vm.deprecatedVMEntryGlobalObject(lexicalGlobalObject);
            RootObject* originRootObject = findRootObject(originGlobalObject);

            JSGlobalObject* globalObject = 0;
            if (value.isObject() && asObject(value)->isGlobalObject())
                globalObject = jsCast<JSGlobalObject*>(asObject(value));

            if (!globalObject)
                globalObject = originGlobalObject;
                
            RootObject* rootObject = findRootObject(globalObject);
            result.objectValue = rootObject
                ? (__bridge CFTypeRef)[webScriptObjectClass() _convertValueToObjcValue:value originRootObject:originRootObject rootObject:rootObject]
                : nil;
        }
        break;

        case ObjcCharType:
        case ObjcUnsignedCharType:
            result.charValue = (char)d;
            break;
        case ObjcShortType:
        case ObjcUnsignedShortType:
            result.shortValue = (short)d;
            break;
        case ObjcBoolType:
            result.booleanValue = (bool)d;
            break;
        case ObjcIntType:
        case ObjcUnsignedIntType:
            result.intValue = (int)d;
            break;
        case ObjcLongType:
        case ObjcUnsignedLongType:
            result.longValue = (long)d;
            break;
        case ObjcLongLongType:
        case ObjcUnsignedLongLongType:
            result.longLongValue = convertDoubleToLongLong(d);
            break;
        case ObjcFloatType:
            result.floatValue = (float)d;
            break;
        case ObjcDoubleType:
            result.doubleValue = (double)d;
            break;
        case ObjcVoidType:
            zeroBytes(result);
            break;

        case ObjcInvalidType:
        default:
            // FIXME: throw an exception?
            break;
    }

    return result;
}

JSValue convertNSStringToString(JSGlobalObject* lexicalGlobalObject, NSString *nsstring)
{
    VM& vm = lexicalGlobalObject->vm();
    JSLockHolder lock(vm);
    JSValue aValue = jsString(vm, String(nsstring));
    return aValue;
}

/*
    ObjC      to    JavaScript
    ----            ----------
    char            number
    short           number
    int             number
    long            number
    float           number
    double          number
    NSNumber        boolean or number
    NSString        string
    NSArray         array
    NSNull          null
    WebScriptObject underlying JavaScript object
    WebUndefined    undefined
    id              object wrapper
    other           should not happen
*/
JSValue convertObjcValueToValue(JSGlobalObject* lexicalGlobalObject, void* buffer, ObjcValueType type, RootObject* rootObject)
{
    JSLockHolder lock(lexicalGlobalObject);
    
    switch (type) {
        case ObjcObjectType: {
            id obj = *(const id*)buffer;
            if (auto *str = dynamic_objc_cast<NSString>(obj))
                return convertNSStringToString(lexicalGlobalObject, str);
            if ([obj isKindOfClass:webUndefinedClass()])
                return jsUndefined();
            if ((__bridge CFBooleanRef)obj == kCFBooleanTrue)
                return jsBoolean(true);
            if ((__bridge CFBooleanRef)obj == kCFBooleanFalse)
                return jsBoolean(false);
            if ([obj isKindOfClass:[NSNumber class]])
                return jsNumber([obj doubleValue]);
            if ([obj isKindOfClass:[NSArray class]])
                return RuntimeArray::create(lexicalGlobalObject, new ObjcArray(obj, rootObject));
            if ([obj isKindOfClass:webScriptObjectClass()]) {
                JSObject* imp = [obj _imp];
                return imp ? imp : jsUndefined();
            }
            if ([obj isKindOfClass:[NSNull class]])
                return jsNull();
            if (obj == 0)
                return jsUndefined();
            return ObjcInstance::create(obj, rootObject)->createRuntimeObject(lexicalGlobalObject);
        }
        case ObjcCharType:
            return jsNumber(*(char*)buffer);
        case ObjcUnsignedCharType:
            return jsNumber(*(unsigned char*)buffer);
        case ObjcShortType:
            return jsNumber(*(short*)buffer);
        case ObjcUnsignedShortType:
            return jsNumber(*(unsigned short*)buffer);
        case ObjcBoolType:
            return jsBoolean(*(bool*)buffer);
        case ObjcIntType:
            return jsNumber(*(int*)buffer);
        case ObjcUnsignedIntType:
            return jsNumber(*(unsigned int*)buffer);
        case ObjcLongType:
            return jsNumber(*(long*)buffer);
        case ObjcUnsignedLongType:
            return jsNumber(*(unsigned long*)buffer);
        case ObjcLongLongType:
            return jsNumber(*(long long*)buffer);
        case ObjcUnsignedLongLongType:
            return jsNumber(*(unsigned long long*)buffer);
        case ObjcFloatType:
            return jsNumber(purifyNaN(*(float*)buffer));
        case ObjcDoubleType:
            return jsNumber(purifyNaN(*(double*)buffer));
        default:
            // Should never get here. Argument types are filtered.
            fprintf(stderr, "%s: invalid type (%d)\n", __PRETTY_FUNCTION__, (int)type);
            ASSERT_NOT_REACHED();
    }
    
    return jsUndefined();
}

ObjcValueType objcValueTypeForType(const char* rawType)
{
    ObjcValueType objcValueType = ObjcInvalidType;

    for (char typeChar : unsafeSpan(rawType)) {
        switch (typeChar) {
            case _C_CONST:
            case _C_BYCOPY:
            case _C_BYREF:
            case _C_ONEWAY:
            case _C_GCINVISIBLE:
                // skip these type modifiers
                break;
            case _C_ID:
                objcValueType = ObjcObjectType;
                break;
            case _C_CHR:
                objcValueType = ObjcCharType;
                break;
            case _C_UCHR:
                objcValueType = ObjcUnsignedCharType;
                break;
            case _C_SHT:
                objcValueType = ObjcShortType;
                break;
            case _C_USHT:
                objcValueType = ObjcUnsignedShortType;
                break;
            case _C_BOOL:
                objcValueType = ObjcBoolType;
                break;
            case _C_INT:
                objcValueType = ObjcIntType;
                break;
            case _C_UINT:
                objcValueType = ObjcUnsignedIntType;
                break;
            case _C_LNG:
                objcValueType = ObjcLongType;
                break;
            case _C_ULNG:
                objcValueType = ObjcUnsignedLongType;
                break;
            case _C_LNG_LNG:
                objcValueType = ObjcLongLongType;
                break;
            case _C_ULNG_LNG:
                objcValueType = ObjcUnsignedLongLongType;
                break;
            case _C_FLT:
                objcValueType = ObjcFloatType;
                break;
            case _C_DBL:
                objcValueType = ObjcDoubleType;
                break;
            case _C_VOID:
                objcValueType = ObjcVoidType;
                break;
            default:
                // Unhandled type. We don't handle C structs, unions, etc.
                // FIXME: throw an exception?
                WTFLogAlways("Unhandled ObjC type specifier: \"%c\" used in ObjC bridge.", typeChar);
                RELEASE_ASSERT_NOT_REACHED();
        }

        if (objcValueType != ObjcInvalidType)
            break;
    }

    return objcValueType;
}

Exception *throwError(JSGlobalObject* lexicalGlobalObject, ThrowScope& scope, NSString *message)
{
    ASSERT(message);
    return throwException(lexicalGlobalObject, scope, JSC::createError(lexicalGlobalObject, String(message)));
}

}
}
