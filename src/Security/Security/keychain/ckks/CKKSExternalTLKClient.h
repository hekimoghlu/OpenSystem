/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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

#if __OBJC2__

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

extern NSString* CKKSSEViewPTA;
extern NSString* CKKSSEViewPTC;

// Note:
//  All fields in these types will be uploaded in plaintext.
//  You _must_ provide your own cryptographic protection of the content of these fields.
@interface CKKSExternalKey : NSObject <NSSecureCoding>

@property (readonly) NSString* view;
@property (readonly) NSString* uuid;
@property (readonly) NSString* parentKeyUUID;
@property (readonly) NSData* keyData;

- (instancetype)initWithView:(NSString*)view
                        uuid:(NSString*)uuid
               parentTLKUUID:(NSString* _Nullable)parentKeyUUID
                     keyData:(NSData*)keyData;

- (NSDictionary*)jsonDictionary;
+ (CKKSExternalKey* _Nullable)parseFromJSONDict:(NSDictionary*)jsonDict error:(NSError**)error;
@end

@interface CKKSExternalTLKShare : NSObject <NSSecureCoding>
@property (readonly) NSString* view;
@property (readonly) NSString* tlkUUID;

@property (readonly) NSData* receiverPeerID;
@property (readonly) NSData* senderPeerID;

@property (nullable, readonly) NSData* wrappedTLK;
@property (nullable, readonly) NSData* signature;

- (instancetype)initWithView:(NSString*)view
                     tlkUUID:(NSString*)tlkUUID
              receiverPeerID:(NSData*)receiverPeerID
                senderPeerID:(NSData*)senderPeerID
                  wrappedTLK:(NSData*)wrappedTLK
                   signature:(NSData*)signature;

- (NSString*)stringifyPeerID:(NSData*)peerID;

- (NSDictionary*)jsonDictionary;
+ (CKKSExternalTLKShare* _Nullable)parseFromJSONDict:(NSDictionary*)jsonDict error:(NSError**)error;
@end

NS_ASSUME_NONNULL_END

#endif // __OBJC2
