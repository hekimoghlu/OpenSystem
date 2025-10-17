/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
//  CKKSPBFileStorage.h
//

#import <Foundation/Foundation.h>
#import <ProtocolBuffer/PBCodable.h>

NS_ASSUME_NONNULL_BEGIN


@protocol CKKSPBCodable <NSObject>
@property (nonatomic, readonly) NSData *data;
+ (instancetype)alloc;
- (id)initWithData:(NSData*)data;
@end

@interface CKKSPBFileStorage<__covariant CKKSConfigurationStorageType : PBCodable *> : NSObject

- (CKKSPBFileStorage *)initWithStoragePath:(NSURL *)storageFile
                              storageClass:(Class<CKKSPBCodable>)storageClass;

- (CKKSConfigurationStorageType _Nullable)storage;
- (void)setStorage:(CKKSConfigurationStorageType _Nonnull)storage;
@end

@interface PBCodable () <CKKSPBCodable>
@end

NS_ASSUME_NONNULL_END
