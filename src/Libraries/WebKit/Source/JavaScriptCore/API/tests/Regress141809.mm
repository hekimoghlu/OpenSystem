/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 14, 2024.
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
#import "Regress141809.h"

#import <objc/objc.h>
#import <objc/runtime.h>

#if JSC_OBJC_API_ENABLED

extern "C" void checkResult(NSString *description, bool passed);
extern "C" void JSSynchronousGarbageCollectForDebugging(JSContextRef);

@protocol TestClassAExports <JSExport>
@end

@interface TestClassA : NSObject<TestClassAExports>
@end

@implementation TestClassA
@end

@protocol TestClassBExports <JSExport>
- (NSString *)name;
@end

@interface TestClassB : TestClassA <TestClassBExports>
@end

@implementation TestClassB
- (NSString *)name
{
    return @"B";
}
@end

@protocol TestClassCExports <JSExport>
- (NSString *)name;
@end

@interface TestClassC : TestClassB <TestClassCExports>
@end

@implementation TestClassC
- (NSString *)name
{
    return @"C";
}
@end

void runRegress141809()
{
    // Test that the ObjC API can correctly re-construct the synthesized
    // prototype and constructor of JS exported ObjC classes.
    // See <https://webkit.org/b/141809>
    @autoreleasepool {
        JSContext *context = [[JSContext alloc] init];
        context[@"print"] = ^(NSString* str) {
            NSLog(@"%@", str);
        };
        
        [context evaluateScript:@"function dumpPrototypes(obj) { \
            var objDepth = 0; \
            var currObj = obj; \
            var objChain = ''; \
            do { \
                var propIndex = 0; \
                var props = ''; \
                Object.getOwnPropertyNames(currObj).forEach(function(val, idx, array) { \
                    props += ((propIndex > 0 ? ', ' : '') + val); \
                    propIndex++; \
                }); \
                var str = ''; \
                if (!objDepth) \
                    str += 'obj '; \
                else { \
                    for (i = 0; i < objDepth; i++) \
                        str += ' '; \
                    str += '--> proto '; \
                } \
                str += currObj; \
                if (props) \
                    str += (' with ' + propIndex + ' props: ' + props); \
                print(str); \
                objChain += (str + '\\n'); \
                objDepth++; \
                currObj = Object.getPrototypeOf(currObj); \
            } while (currObj); \
            return { objDepth: objDepth, objChain: objChain }; \
        }"];
        JSValue* dumpPrototypes = context[@"dumpPrototypes"];
        
        JSValue* resultBeforeGC = nil;
        @autoreleasepool {
            TestClassC* obj = [[TestClassC alloc] init];
            resultBeforeGC = [dumpPrototypes callWithArguments:@[obj]];
        }
        
        JSSynchronousGarbageCollectForDebugging([context JSGlobalContextRef]);
        
        @autoreleasepool {
            TestClassC* obj = [[TestClassC alloc] init];
            JSValue* resultAfterGC = [dumpPrototypes callWithArguments:@[obj]];
            checkResult(@"object and prototype chain depth is 5 deep", [resultAfterGC[@"objDepth"] toInt32] == 5);
            checkResult(@"object and prototype chain depth before and after GC matches", [resultAfterGC[@"objDepth"] toInt32] == [resultBeforeGC[@"objDepth"] toInt32]);
            checkResult(@"object and prototype chain before and after GC matches", [[resultAfterGC[@"objChain"] toString] isEqualToString:[resultBeforeGC[@"objChain"] toString]]);
        }
    }
}

#endif // JSC_OBJC_API_ENABLED
