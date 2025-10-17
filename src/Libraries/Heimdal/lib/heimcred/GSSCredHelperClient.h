/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 2, 2023.
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
#ifndef GSSCredHelperClient_h
#define GSSCredHelperClient_h

#import <Foundation/Foundation.h>
#import "krb5.h"
#import "heimcred.h"

NS_ASSUME_NONNULL_BEGIN

@protocol GSSCredHelperClient <NSObject>

+ (krb5_error_code)acquireForCred:(HeimCredRef)cred expireTime:(time_t *)expire;
+ (krb5_error_code)refreshForCred:(HeimCredRef)cred expireTime:(time_t *)expire;

@end

NS_ASSUME_NONNULL_END


#endif /* GSSCredHelperClient_h */
