/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#import "JSWrapperMapTests.h"

#import "APICast.h"
#import "HeapCellInlines.h"
#import "JSGlobalObjectInlines.h"
#import "JSValue.h"

#if JSC_OBJC_API_ENABLED

extern "C" void checkResult(NSString *description, bool passed);

@protocol TestClassJSExport <JSExport>
- (instancetype)init;
@end

@interface TestClass : NSObject <TestClassJSExport>
@end

@implementation TestClass
@end


@interface JSWrapperMapTests : NSObject
+ (void)testStructureIdentity;
@end


@implementation JSWrapperMapTests
+ (void)testStructureIdentity
{
    JSContext* context = [[JSContext alloc] init];
    JSGlobalContextRef contextRef = JSGlobalContextRetain(context.JSGlobalContextRef);
    JSC::JSGlobalObject* globalObject = toJS(contextRef);

    context[@"TestClass"] = [TestClass class];
    JSValue* aWrapper = [context evaluateScript:@"new TestClass()"];
    JSValue* bWrapper = [context evaluateScript:@"new TestClass()"];
    JSC::JSValue aValue = toJS(globalObject, aWrapper.JSValueRef);
    JSC::JSValue bValue = toJS(globalObject, bWrapper.JSValueRef);
    JSC::Structure* aStructure = aValue.structureOrNull();
    JSC::Structure* bStructure = bValue.structureOrNull();
    checkResult(@"structure should not be null", !!aStructure);
    checkResult(@"both wrappers should share the same structure", aStructure == bStructure);
}
@end

void runJSWrapperMapTests()
{
    @autoreleasepool {
        [JSWrapperMapTests testStructureIdentity];
    }
}

#endif // JSC_OBJC_API_ENABLED
