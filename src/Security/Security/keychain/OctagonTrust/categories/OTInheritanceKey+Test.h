/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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
#ifndef OTInheritanceKeyTest_h
#define OTInheritanceKeyTest_h

#if __OBJC2__

#import <Foundation/Foundation.h>
#import "keychain/OctagonTrust/OTInheritanceKey.h"

NS_ASSUME_NONNULL_BEGIN

@interface OTInheritanceKey (Test)
+ (NSString* _Nullable)base32:(const unsigned char*)d len:(size_t)inlen;
+ (NSData* _Nullable)unbase32:(const unsigned char*)s len:(size_t)inlen;
+ (NSString* _Nullable)printableWithData:(NSData*)data checksumSize:(size_t)checksumSize error:(NSError**)error;
+ (NSData* _Nullable)parseBase32:(NSString*)in checksumSize:(size_t)checksumSize error:(NSError**)error;
@end

NS_ASSUME_NONNULL_END

#endif /* OBJC2 */

#endif /* OTInheritanceKeyTest_h */
