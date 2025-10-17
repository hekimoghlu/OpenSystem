/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
//  NSError+KCCreationHelpers.h
//  KeychainCircle
//
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Returns false and fills in error with formatted description if cc_result is an error
bool CoreCryptoError(int cc_result, NSError * _Nullable * _Nullable error,  NSString * _Nonnull description, ...) NS_FORMAT_FUNCTION(3, 4);
// Returns false and fills in a requirement error if requirement is false
// We should have something better than -50 here.
bool RequirementError(bool requirement, NSError * _Nullable * _Nullable error, NSString * _Nonnull description, ...) NS_FORMAT_FUNCTION(3, 4);

bool OSStatusError(OSStatus status, NSError * _Nullable * _Nullable error, NSString* _Nonnull description, ...) NS_FORMAT_FUNCTION(3, 4);


// MARK: Error Extensions
@interface NSError(KCCreationHelpers)

+ (instancetype) errorWithOSStatus:(OSStatus) status
                          userInfo:(NSDictionary *)dict;

- (instancetype) initWithOSStatus:(OSStatus) status
                         userInfo:(NSDictionary *)dict;

+ (instancetype) errorWithOSStatus:(OSStatus) status
                       description:(NSString*)description
                              args:(va_list)va NS_FORMAT_FUNCTION(2, 0);

- (instancetype) initWithOSStatus:(OSStatus) status
                      description:(NSString*)description
                             args:(va_list)va NS_FORMAT_FUNCTION(2, 0);

+ (instancetype) errorWithCoreCryptoStatus:(int) status
                                  userInfo:(NSDictionary *)dict;

- (instancetype) initWithCoreCryptoStatus:(int) status
                                 userInfo:(NSDictionary *)dict;

+ (instancetype) errorWithCoreCryptoStatus:(int) status
                               description:(NSString*)description
                                      args:(va_list)va NS_FORMAT_FUNCTION(2, 0);

- (instancetype) initWithCoreCryptoStatus:(int) status
                              description:(NSString*)description
                                     args:(va_list)va NS_FORMAT_FUNCTION(2, 0);

@end

NS_ASSUME_NONNULL_END
