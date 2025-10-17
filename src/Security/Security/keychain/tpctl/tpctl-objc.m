/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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


#import "tpctl-objc.h"

@implementation TPCTLObjectiveC

+ (BOOL)catchNSException:(void(^)(void))block error:(NSError**)error {
    @try {
        block();
        return true;
    }
    @catch(NSException* exception) {
        if(error) {
            NSMutableDictionary* ui = exception.userInfo ? [exception.userInfo mutableCopy] : [NSMutableDictionary dictionary];
            if(exception.reason) {
                ui[NSLocalizedDescriptionKey] = exception.reason;
            }
            *error = [NSError errorWithDomain:exception.name code:0 userInfo:ui];
        }
        return false;
    }
}

+ (NSString* _Nullable)jsonSerialize:(id)something error:(NSError**)error {
    @try {
        NSError* localError = nil;
        NSData* jsonData = [NSJSONSerialization dataWithJSONObject:something options:(NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys) error:&localError];
        if(!jsonData || localError) {
            if(error) {
                *error = localError;
            }
            return nil;
        }

        NSString* utf8String = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
        if(!utf8String) {
            if(error) {
                *error = [NSError errorWithDomain:@"text" code:0 userInfo:@{NSLocalizedDescriptionKey: @"JSON data could not be decoded as UTF8"}];
            }
            return nil;
        }
        return utf8String;
    }
    @catch(NSException* exception) {
        if(error) {
            NSMutableDictionary* ui = exception.userInfo ? [exception.userInfo mutableCopy] : [NSMutableDictionary dictionary];
            if(exception.reason) {
                ui[NSLocalizedDescriptionKey] = exception.reason;
            }
            *error = [NSError errorWithDomain:exception.name code:0 userInfo:ui];
        }
        return nil;
    }
}

@end
