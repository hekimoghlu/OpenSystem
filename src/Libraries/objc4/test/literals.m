/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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

// TEST_CONFIG OS=!exclavekit LANGUAGE=objc,objc++
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import <Foundation/NSDictionary.h>
#import <objc/runtime.h>
#import <objc/objc-abi.h>
#import <math.h>
#include "test.h"

int main() {
    PUSH_POOL {

#if __has_feature(objc_bool)    // placeholder until we get a more precise macro.
        NSArray *array = @[ @1, @2, @YES, @NO, @"Hello", @"World" ];
        testassert([array count] == 6);
        NSDictionary *dict = @{ @"Name" : @"John Q. Public", @"Age" : @42 };
        testassert([dict count] == 2);
        NSDictionary *numbers = @{ @"Ï€" : @M_PI, @"e" : @M_E };
        testassert([[numbers objectForKey:@"Ï€"] doubleValue] == M_PI);
        testassert([[numbers objectForKey:@"e"] doubleValue] == M_E);

        BOOL yesBool = YES;
        BOOL noBool = NO;
        array = @[
            @(true),
            @(YES),
            [NSNumber numberWithBool:YES],
            @YES,
            @(yesBool),
            @((BOOL)YES),
        
            @(false),
            @(NO),
            [NSNumber numberWithBool:NO],
            @NO,
            @(noBool),
            @((BOOL)NO),
        ];
        NSData * jsonData = [NSJSONSerialization dataWithJSONObject:array options:0 error:nil];
        NSString * string = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
#if __cplusplus
        testassert([string isEqualToString:@"[true,true,true,true,true,true,false,false,false,false,false,false]"]);
#else
        // C99 @(true) and @(false) evaluate to @(1) and @(0).
        testassert([string isEqualToString:@"[1,true,true,true,true,true,0,false,false,false,false,false]"]);
#endif

#endif
        
    } POP_POOL;

    succeed(__FILE__);

    return 0;
}
