/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
#import "objc_class.h"

#import "WebScriptObject.h"
#import "WebScriptObjectProtocol.h"
#import "objc_instance.h"
#import <JavaScriptCore/JSGlobalObjectInlines.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/ObjCRuntimeExtras.h>
#import <wtf/RetainPtr.h>

namespace JSC {
namespace Bindings {

ObjcClass::ObjcClass(ClassStructPtr aClass)
    : _isa(aClass)
{
}

static RetainPtr<CFMutableDictionaryRef>& classesByIsA()
{
    static NeverDestroyed<RetainPtr<CFMutableDictionaryRef>> classesByIsA;
    return classesByIsA;
}

static void _createClassesByIsAIfNecessary()
{
    if (!classesByIsA())
        classesByIsA() = adoptCF(CFDictionaryCreateMutable(NULL, 0, NULL, NULL));
}

ObjcClass* ObjcClass::classForIsA(ClassStructPtr isa)
{
    _createClassesByIsAIfNecessary();

    auto aClass = reinterpret_cast<ObjcClass*>(const_cast<void*>(CFDictionaryGetValue(classesByIsA().get(), (__bridge CFTypeRef)isa)));
    if (!aClass) {
        aClass = new ObjcClass(isa);
        CFDictionaryAddValue(classesByIsA().get(), (__bridge CFTypeRef)isa, aClass);
    }

    return aClass;
}

/*
    By default, a JavaScript method name is produced by concatenating the
    components of an ObjectiveC method name, replacing ':' with '_', and
    escaping '_' and '$' with a leading '$', such that '_' becomes "$_" and
    '$' becomes "$$". For example:

    ObjectiveC name         Default JavaScript name
        moveTo::                moveTo__
        moveTo_                 moveTo$_
        moveTo$_                moveTo$$$_

    This function performs the inverse of that operation.

    @result Fills 'buffer' with the ObjectiveC method name that corresponds to 'JSName'.
*/
typedef Vector<char, 256> JSNameConversionBuffer;
static inline void convertJSMethodNameToObjc(const CString& jsName, JSNameConversionBuffer& buffer)
{
    auto characters = jsName.unsafeSpanIncludingNullTerminator();
    buffer.reserveInitialCapacity(characters.size());
    for (size_t i = 0; i < characters.size(); ++i) {
        if (characters[i] == '$') {
            ++i;
            buffer.append(characters[i]);
        } else if (characters[i] == '_')
            buffer.append(':');
        else
            buffer.append(characters[i]);
    }
}

Method* ObjcClass::methodNamed(PropertyName propertyName, Instance*) const
{
    String name(propertyName.publicName());
    if (name.isNull())
        return nullptr;

    if (Method* method = m_methodCache.get(name.impl()))
        return method;

    CString jsName = name.ascii();
    JSNameConversionBuffer buffer;
    convertJSMethodNameToObjc(jsName, buffer);
    RetainPtr<NSString> methodName = adoptNS([[NSString alloc] initWithCString:buffer.data() encoding:NSASCIIStringEncoding]);

    Method* methodPtr = 0;
    ClassStructPtr thisClass = _isa;
    
    while (thisClass && !methodPtr) {
        auto objcMethodList = class_copyMethodListSpan(thisClass);
        for (auto& objcMethod : objcMethodList.span()) {
            SEL objcMethodSelector = method_getName(objcMethod);
            const char* objcMethodSelectorName = sel_getName(objcMethodSelector);
            NSString* mappedName = nil;

            // See if the class wants to exclude the selector from visibility in JavaScript.
            if ([thisClass respondsToSelector:@selector(isSelectorExcludedFromWebScript:)])
                if ([thisClass isSelectorExcludedFromWebScript:objcMethodSelector])
                    continue;

            // See if the class want to provide a different name for the selector in JavaScript.
            // Note that we do not do any checks to guarantee uniqueness. That's the responsiblity
            // of the class.
            if ([thisClass respondsToSelector:@selector(webScriptNameForSelector:)])
                mappedName = [thisClass webScriptNameForSelector:objcMethodSelector];

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
            if ((mappedName && [mappedName isEqual:methodName.get()]) || !strcmp(objcMethodSelectorName, buffer.data())) {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
                auto method = makeUnique<ObjcMethod>(thisClass, objcMethodSelector);
                methodPtr = method.get();
                m_methodCache.add(name.impl(), WTFMove(method));
                break;
            }
        }
        thisClass = class_getSuperclass(thisClass);
    }

    return methodPtr;
}

Field* ObjcClass::fieldNamed(PropertyName propertyName, Instance* instance) const
{
    String name(propertyName.publicName());
    if (name.isNull())
        return nullptr;

    Field* field = m_fieldCache.get(name.impl());
    if (field)
        return field;

    ClassStructPtr thisClass = _isa;

    CString jsName = name.ascii();
    RetainPtr<NSString> fieldName = adoptNS([[NSString alloc] initWithCString:jsName.data() encoding:NSASCIIStringEncoding]);
    id targetObject = (static_cast<ObjcInstance*>(instance))->getObject();
#if PLATFORM(IOS_FAMILY)
    IGNORE_WARNINGS_BEGIN("undeclared-selector")
    id attributes = [targetObject respondsToSelector:@selector(attributeKeys)] ? [targetObject performSelector:@selector(attributeKeys)] : nil;
    IGNORE_WARNINGS_END
#else
    id attributes = [targetObject attributeKeys];
#endif
    if (attributes) {
        // Class overrides attributeKeys, use that array of key names.
        for (NSString* keyName in attributes) {
            const char* UTF8KeyName = [keyName UTF8String]; // ObjC actually only supports ASCII names.

            // See if the class wants to exclude the selector from visibility in JavaScript.
            if ([thisClass respondsToSelector:@selector(isKeyExcludedFromWebScript:)])
                if ([thisClass isKeyExcludedFromWebScript:UTF8KeyName])
                    continue;

            // See if the class want to provide a different name for the selector in JavaScript.
            // Note that we do not do any checks to guarantee uniqueness. That's the responsiblity
            // of the class.
            NSString* mappedName = nil;
            if ([thisClass respondsToSelector:@selector(webScriptNameForKey:)])
                mappedName = [thisClass webScriptNameForKey:UTF8KeyName];

            if ((mappedName && [mappedName isEqual:fieldName.get()]) || [keyName isEqual:fieldName.get()]) {
                auto newField = makeUnique<ObjcField>((__bridge CFStringRef)keyName);
                field = newField.get();
                m_fieldCache.add(name.impl(), WTFMove(newField));
                break;
            }
        }
    } else {
        // Class doesn't override attributeKeys, so fall back on class runtime
        // introspection.

        while (thisClass) {
            auto ivarsInClass = class_copyIvarListSpan(thisClass);
            for (auto& objcIVar : ivarsInClass.span()) {
                const char* objcIvarName = ivar_getName(objcIVar);
                NSString *mappedName = nullptr;

                // See if the class wants to exclude the selector from visibility in JavaScript.
                if ([thisClass respondsToSelector:@selector(isKeyExcludedFromWebScript:)])
                    if ([thisClass isKeyExcludedFromWebScript:objcIvarName])
                        continue;

                // See if the class want to provide a different name for the selector in JavaScript.
                // Note that we do not do any checks to guarantee uniqueness. That's the responsiblity
                // of the class.
                if ([thisClass respondsToSelector:@selector(webScriptNameForKey:)])
                    mappedName = [thisClass webScriptNameForKey:objcIvarName];

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
                if ((mappedName && [mappedName isEqual:fieldName.get()]) || !strcmp(objcIvarName, jsName.data())) {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
                    auto newField = makeUnique<ObjcField>(objcIVar);
                    field = newField.get();
                    m_fieldCache.add(name.impl(), WTFMove(newField));
                    break;
                }
            }

            thisClass = class_getSuperclass(thisClass);
        }
    }

    return field;
}

JSValue ObjcClass::fallbackObject(JSGlobalObject* lexicalGlobalObject, Instance* instance, PropertyName propertyName)
{
    ObjcInstance* objcInstance = static_cast<ObjcInstance*>(instance);
    id targetObject = objcInstance->getObject();
    
    if (![targetObject respondsToSelector:@selector(invokeUndefinedMethodFromWebScript:withArguments:)])
        return jsUndefined();

    if (!propertyName.publicName())
        return jsUndefined();

    return ObjcFallbackObjectImp::create(lexicalGlobalObject, lexicalGlobalObject, objcInstance, propertyName.publicName());
}

}
}
