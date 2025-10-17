/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#if OCTAGON

#include "keychain/securityd/SecDbItem.h"

@class CKKSItem;
@class CKKSMirrorEntry;
@class CKKSKey;
@class CKKSMemoryKeyCache;
@class CKKSOutgoingQueueEntry;
@class CKKSAESSIVKey;
@class CKRecordZoneID;

NS_ASSUME_NONNULL_BEGIN

#define CKKS_PADDING_MARK_BYTE 0x80

@interface CKKSItemEncrypter : NSObject

+ (CKKSItem* _Nullable)encryptCKKSItem:(CKKSItem*)baseitem
                        dataDictionary:(NSDictionary*)dict
                      updatingCKKSItem:(CKKSItem* _Nullable)olditem
                             parentkey:(CKKSKey*)parentkey
                              keyCache:(CKKSMemoryKeyCache* _Nullable)keyCache
                                 error:(NSError* _Nullable __autoreleasing* _Nullable)error;

+ (NSDictionary* _Nullable)decryptItemToDictionary:(CKKSItem*)item
                                          keyCache:(CKKSMemoryKeyCache* _Nullable)keyCache
                                             error:(NSError* _Nullable __autoreleasing* _Nullable)error;

+ (NSData* _Nullable)encryptDictionary:(NSDictionary*)dict
                                   key:(CKKSAESSIVKey*)key
                     authenticatedData:(NSDictionary<NSString*, NSData*>* _Nullable)ad
                                 error:(NSError* _Nullable __autoreleasing* _Nullable)error;
+ (NSDictionary* _Nullable)decryptDictionary:(NSData*)encitem
                                         key:(CKKSAESSIVKey*)key
                           authenticatedData:(NSDictionary<NSString*, NSData*>* _Nullable)ad
                                       error:(NSError* _Nullable __autoreleasing* _Nullable)error;

+ (NSData*)padData:(NSData*)input blockSize:(NSUInteger)blockSize additionalBlock:(BOOL)extra;
+ (NSData* _Nullable)removePaddingFromData:(NSData*)input;
@end

NS_ASSUME_NONNULL_END
#endif  // OCTAGON
