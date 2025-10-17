/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#import "JSExportTests.h"

#import <objc/runtime.h>
#import <objc/objc.h>

#if JSC_OBJC_API_ENABLED

extern "C" void checkResult(NSString *description, bool passed);

@interface JSExportTests : NSObject
+ (void) exportInstanceMethodWithIdProtocolTest;
+ (void) exportInstanceMethodWithClassProtocolTest;
+ (void) exportDynamicallyGeneratedProtocolTest;
@end

@protocol TruthTeller
- (BOOL) returnTrue;
@end

@interface TruthTeller : NSObject<TruthTeller>
@end

@implementation TruthTeller
- (BOOL) returnTrue
{
    return true;
}
@end

@protocol ExportMethodWithIdProtocol <JSExport>
- (void) methodWithIdProtocol:(id<TruthTeller>)object;
@end

@interface ExportMethodWithIdProtocol : NSObject<ExportMethodWithIdProtocol>
@end

@implementation ExportMethodWithIdProtocol
- (void) methodWithIdProtocol:(id<TruthTeller>)object
{
    checkResult(@"Exporting a method with id<protocol> in the type signature", [object returnTrue]);
}
@end

@protocol ExportMethodWithClassProtocol <JSExport>
- (void) methodWithClassProtocol:(NSObject<TruthTeller> *)object;
@end

@interface ExportMethodWithClassProtocol : NSObject<ExportMethodWithClassProtocol>
@end

@implementation ExportMethodWithClassProtocol
- (void) methodWithClassProtocol:(NSObject<TruthTeller> *)object
{
    checkResult(@"Exporting a method with class<protocol> in the type signature", [object returnTrue]);
}
@end

@interface NoUnderscorePrefix : NSObject
@end

@implementation NoUnderscorePrefix
@end

@interface _UnderscorePrefixNoExport : NoUnderscorePrefix
@end

@implementation _UnderscorePrefixNoExport
@end

@protocol Initializing <JSExport>
- (instancetype)init;
@end

@interface _UnderscorePrefixWithExport : NoUnderscorePrefix <Initializing>
@end

@implementation _UnderscorePrefixWithExport
@end

@implementation JSExportTests
+ (void) exportInstanceMethodWithIdProtocolTest
{
    JSContext *context = [[JSContext alloc] init];
    context[@"ExportMethodWithIdProtocol"] = [ExportMethodWithIdProtocol class];
    context[@"makeTestObject"] = ^{
        return [[ExportMethodWithIdProtocol alloc] init];
    };
    context[@"opaqueObject"] = [[TruthTeller alloc] init];
    [context evaluateScript:@"makeTestObject().methodWithIdProtocol(opaqueObject);"];
    checkResult(@"Successfully exported instance method", !context.exception);
}

+ (void) exportInstanceMethodWithClassProtocolTest
{
    JSContext *context = [[JSContext alloc] init];
    context[@"ExportMethodWithClassProtocol"] = [ExportMethodWithClassProtocol class];
    context[@"makeTestObject"] = ^{
        return [[ExportMethodWithClassProtocol alloc] init];
    };
    context[@"opaqueObject"] = [[TruthTeller alloc] init];
    [context evaluateScript:@"makeTestObject().methodWithClassProtocol(opaqueObject);"];
    checkResult(@"Successfully exported instance method", !context.exception);
}

+ (void) exportDynamicallyGeneratedProtocolTest
{
    JSContext *context = [[JSContext alloc] init];
    Protocol *dynProtocol = objc_allocateProtocol("NSStringJSExport");
    Protocol *jsExportProtocol = @protocol(JSExport);
    protocol_addProtocol(dynProtocol, jsExportProtocol);
    Method method = class_getInstanceMethod([NSString class], @selector(boolValue));
    protocol_addMethodDescription(dynProtocol, @selector(boolValue), method_getTypeEncoding(method), YES, YES);
    NSLog(@"type encoding = %s", method_getTypeEncoding(method));
    protocol_addMethodDescription(dynProtocol, @selector(boolValue), "B@:", YES, YES);
    objc_registerProtocol(dynProtocol);
    class_addProtocol([NSString class], dynProtocol);
    
    context[@"NSString"] = [NSString class];
    context[@"myString"] = @"YES";
    JSValue *value = [context evaluateScript:@"myString.boolValue()"];
    checkResult(@"Dynamically generated JSExport-ed protocols are ignored", [value isUndefined] && !!context.exception);
}

+ (void)classNamePrefixedWithUnderscoreTest
{
    JSContext *context = [[JSContext alloc] init];

    context[@"_UnderscorePrefixNoExport"] = [_UnderscorePrefixNoExport class];
    context[@"_UnderscorePrefixWithExport"] = [_UnderscorePrefixWithExport class];

    checkResult(@"Non-underscore-prefixed ancestor class used when there are no exports", [context[@"_UnderscorePrefixNoExport"] toObject] == [NoUnderscorePrefix class]);
    checkResult(@"Underscore-prefixed class used when there are exports", [context[@"_UnderscorePrefixWithExport"] toObject] == [_UnderscorePrefixWithExport class]);

    JSValue *withExportInstance = [context evaluateScript:@"new _UnderscorePrefixWithExport()"];
    checkResult(@"Exports present on underscore-prefixed class", !context.exception && !withExportInstance.isUndefined);
}

@end

@protocol AJSExport <JSExport>
- (instancetype)init;
@end

@interface A : NSObject <AJSExport>
@end

@implementation A
@end

static void wrapperLifetimeIsTiedToGlobalObject()
{
    JSGlobalContextRef contextRef;
    @autoreleasepool {
        JSContext *context = [[JSContext alloc] init];
        contextRef = JSGlobalContextRetain(context.JSGlobalContextRef);
        context[@"A"] = A.class;
        checkResult(@"Initial wrapper's constructor is itself", [[context evaluateScript:@"new A().constructor === A"] toBool]);
    }

    @autoreleasepool {
        JSContext *context = [JSContext contextWithJSGlobalContextRef:contextRef];
        checkResult(@"New context's wrapper's constructor is itself", [[context evaluateScript:@"new A().constructor === A"] toBool]);
    }

    JSGlobalContextRelease(contextRef);
}

static void wrapperForNSObjectisObject()
{
    @autoreleasepool {
        JSContext *context = [[JSContext alloc] init];
        context[@"Object"] = [[NSNull alloc] init];
        context.exception = nil;

        context[@"A"] = NSObject.class;
        checkResult(@"Should not throw an exception when wrapping NSObject and Object has been changed", ![context exception]);
    }
}

void runJSExportTests()
{
    @autoreleasepool {
        [JSExportTests exportInstanceMethodWithIdProtocolTest];
        [JSExportTests exportInstanceMethodWithClassProtocolTest];
        [JSExportTests exportDynamicallyGeneratedProtocolTest];
        [JSExportTests classNamePrefixedWithUnderscoreTest];
    }
    wrapperLifetimeIsTiedToGlobalObject();
    wrapperForNSObjectisObject();
}

#endif // JSC_OBJC_API_ENABLED
