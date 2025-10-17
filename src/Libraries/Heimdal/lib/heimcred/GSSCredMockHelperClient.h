/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
//  GSSCredMockHelperClient.h
//  GSSCredTests
//
//  Created by Matt Chanda on 6/16/20.
//

#import <Foundation/Foundation.h>
#import "GSSCredHelperClient.h"
#import <XCTest/XCTest.h>

NS_ASSUME_NONNULL_BEGIN

@interface GSSCredMockHelperClient : NSObject<GSSCredHelperClient>

typedef krb5_error_code (^clientBlock)(HeimCredRef, time_t *);

@property (nonatomic, class, copy, nullable) clientBlock expireBlock;
@property (nonatomic, class, copy, nullable) clientBlock renewBlock;

@property (nonatomic, nullable)  NSMutableDictionary<NSString *, XCTestExpectation *> *expireExpectations;
@property (nonatomic, nullable)  NSMutableDictionary<NSString *, XCTestExpectation *> *renewExpectations;
@property (nonatomic, nullable)  XCTestExpectation *finalExpectation;

+ (krb5_error_code)acquireForCred:(nonnull HeimCredRef)cred expireTime:(nonnull time_t *)expire;
+ (krb5_error_code)refreshForCred:(nonnull HeimCredRef)cred expireTime:(nonnull time_t *)expire;

+ (void)setExpireBlock:(clientBlock) block;
+ (void)setRenewBlock:(clientBlock) block;

@end

NS_ASSUME_NONNULL_END
