/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
//  AppleKeyboardStateManager.m
//  AppleKeyboardFilter
//
//  Created by Daniel Kim on 3/22/18.
//  Copyright Â© 2018 Apple. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "AppleKeyboardStateManager.h"

@interface AppleKeyboardStateManager()
@property (nonatomic) NSMutableSet <NSNumber *> *capsLockStateTable;
@end

@implementation AppleKeyboardStateManager

+ (instancetype)sharedManager {
    static AppleKeyboardStateManager *sharedManager = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedManager = [[self alloc] init];
    });
    return sharedManager;
}

-(instancetype)init
{
    self = [super init];
    
    if (!self) {
        return self;
    }
    
    _capsLockStateTable = [NSMutableSet new];
    
    return self;
}

- (BOOL)isCapsLockEnabled:(NSNumber *)locationID
{
    BOOL result = NO;
    
    if (!locationID) {
        return result;
    }
    
    @synchronized (self) {
        result = [_capsLockStateTable containsObject:locationID];
    }
    
    return result;
}

- (void)setCapsLockEnabled:(BOOL)enable
                locationID:(NSNumber *)locationID
{
    if (!locationID) {
        return;
    }
    
    @synchronized (self) {
        if (enable) {
            [_capsLockStateTable addObject:locationID];
        } else {
            [_capsLockStateTable removeObject:locationID];
        }
    }
}

@end
