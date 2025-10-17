/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

//
//  KCError.m
//  Security
//
//

#import "KCError.h"

NSString* KCErrorDomain = @"com.apple.security.keychaincircle";

@implementation NSError(KCJoiningError)

+ (nonnull instancetype) errorWithJoiningError:(KCJoiningError) code
                                        format:(NSString*) format
                                     arguments:(va_list) va {
    return [[NSError alloc] initWithJoiningError:code
                                        userInfo:@{NSLocalizedDescriptionKey:[[NSString alloc] initWithFormat:format arguments:va]}];

}

+ (nonnull instancetype) errorWithJoiningError:(KCJoiningError) code
                                        format:(NSString*) format, ... {

    va_list va;
    va_start(va, format);
    NSError* result = [NSError errorWithJoiningError:code format:format arguments:va];
    va_end(va);

    return result;

}
- (nonnull instancetype) initWithJoiningError:(KCJoiningError) code
                                     userInfo:(nonnull NSDictionary *)dict {
    return [self initWithDomain:KCErrorDomain code:code userInfo:dict];
}
@end

void KCJoiningErrorCreate(KCJoiningError code, NSError** error, NSString* format, ...) {
    if (error && (*error == nil)) {
        va_list va;
        va_start(va, format);
        *error = [NSError errorWithJoiningError:code format:format arguments:va];
        va_end(va);
    }
}

