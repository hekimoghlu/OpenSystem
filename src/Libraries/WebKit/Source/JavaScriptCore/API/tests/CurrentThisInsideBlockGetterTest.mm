/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#import "CurrentThisInsideBlockGetterTest.h"

#if JSC_OBJC_API_ENABLED

#import <Foundation/Foundation.h>
#import <JavaScriptCore/JavaScriptCore.h>

static JSObjectRef CallAsConstructor(JSContextRef ctx, JSObjectRef constructor, size_t, const JSValueRef[], JSValueRef*)
{
    JSObjectRef newObjectRef = NULL;
    NSMutableDictionary *constructorPrivateProperties = (__bridge NSMutableDictionary *)(JSObjectGetPrivate(constructor));
    NSDictionary *constructorDescriptor = constructorPrivateProperties[@"constructorDescriptor"];
    newObjectRef = JSObjectMake(ctx, NULL, NULL);
    NSDictionary *objectProperties = constructorDescriptor[@"objectProperties"];
    
    if (objectProperties) {
        JSValue *newObject = [JSValue valueWithJSValueRef:newObjectRef inContext:[JSContext contextWithJSGlobalContextRef:JSContextGetGlobalContext(ctx)]];
        for (NSString *objectProperty in objectProperties) {
            [newObject defineProperty:objectProperty descriptor:objectProperties[objectProperty]];
        }
    }
    
    return newObjectRef;
}

static void ConstructorFinalize(JSObjectRef object)
{
    NSMutableDictionary *privateProperties = (__bridge NSMutableDictionary *)(JSObjectGetPrivate(object));
    CFBridgingRelease((__bridge CFTypeRef)(privateProperties));
    JSObjectSetPrivate(object, NULL);
}

static JSClassRef ConstructorClass(void)
{
    static JSClassRef constructorClass = NULL;
    
    if (constructorClass == NULL) {
        JSClassDefinition classDefinition = kJSClassDefinitionEmpty;
        classDefinition.className = "Constructor";
        classDefinition.callAsConstructor = CallAsConstructor;
        classDefinition.finalize = ConstructorFinalize;
        constructorClass = JSClassCreate(&classDefinition);
    }
    
    return constructorClass;
}

@interface JSValue (ConstructorCreation)

+ (JSValue *)valueWithConstructorDescriptor:(NSDictionary *)constructorDescriptor inContext:(JSContext *)context;

@end

@implementation JSValue (ConstructorCreation)

+ (JSValue *)valueWithConstructorDescriptor:(id)constructorDescriptor inContext:(JSContext *)context
{
    NSMutableDictionary *privateProperties = [@{ @"constructorDescriptor" : constructorDescriptor } mutableCopy];
    JSGlobalContextRef ctx = [context JSGlobalContextRef];
    JSObjectRef constructorRef = JSObjectMake(ctx, ConstructorClass(), const_cast<void*>(CFBridgingRetain(privateProperties)));
    JSValue *constructor = [JSValue valueWithJSValueRef:constructorRef inContext:context];
    return constructor;
}

@end

@interface JSContext (ConstructorCreation)

- (JSValue *)valueWithConstructorDescriptor:(NSDictionary *)constructorDescriptor;

@end

@implementation JSContext (ConstructorCreation)

- (JSValue *)valueWithConstructorDescriptor:(id)constructorDescriptor
{
    return [JSValue valueWithConstructorDescriptor:constructorDescriptor inContext:self];
}

@end

void currentThisInsideBlockGetterTest()
{
    @autoreleasepool {
        JSContext *context = [[JSContext alloc] init];
        
        JSValue *myConstructor = [context valueWithConstructorDescriptor:@{
            @"objectProperties" : @{
                @"currentThis" : @{ JSPropertyDescriptorGetKey : ^{ return JSContext.currentThis; } },
            },
        }];
        
        JSValue *myObj1 = [myConstructor constructWithArguments:nil];
        NSLog(@"myObj1.currentThis: %@", myObj1[@"currentThis"]);
        JSValue *myObj2 = [myConstructor constructWithArguments:@[ @"bar" ]];
        NSLog(@"myObj2.currentThis: %@", myObj2[@"currentThis"]);
    }
}

#endif // JSC_OBJC_API_ENABLED
