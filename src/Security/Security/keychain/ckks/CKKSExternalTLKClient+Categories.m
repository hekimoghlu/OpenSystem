/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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

#import "keychain/ckks/CKKSExternalTLKClient+Categories.h"
#import "keychain/categories/NSError+UsefulConstructors.h"

@implementation CKKSExternalKey (CKKSTranslation)

- (instancetype)initWithViewName:(NSString*)viewName
                             tlk:(CKKSKey*)tlk
{
    return [self initWithView:viewName
                         uuid:tlk.uuid
                parentTLKUUID:tlk.parentKeyUUID
                      keyData:tlk.wrappedKeyData];
}

- (CKKSKey* _Nullable)makeCKKSKey:(CKRecordZoneID*)zoneID
                        contextID:(NSString*)contextID
                            error:(NSError**)error
{
    CKKSKey* key = [[CKKSKey alloc] initWithWrappedKeyData:self.keyData
                                                 contextID:contextID
                                                      uuid:self.uuid
                                             parentKeyUUID:self.parentKeyUUID ?: self.uuid
                                                  keyclass:SecCKKSKeyClassTLK
                                                     state:SecCKKSProcessedStateRemote
                                                    zoneID:zoneID
                                           encodedCKRecord:nil
                                                currentkey:0];

    return key;
}

- (CKKSKey* _Nullable)makeFakeCKKSClassKey:(CKKSKeyClass*)keyclass
                                 contextID:(NSString*)contextID
                                    zoneiD:(CKRecordZoneID*)zoneID
                                     error:(NSError**)error
{
    CKKSKey* key = [[CKKSKey alloc] initWithWrappedKeyData:self.keyData
                                                 contextID:contextID
                                                      uuid:[NSString stringWithFormat:@"%@-fake-%@", self.uuid, keyclass]
                                             parentKeyUUID:self.parentKeyUUID
                                                  keyclass:keyclass
                                                     state:SecCKKSProcessedStateRemote
                                                    zoneID:zoneID
                                           encodedCKRecord:nil
                                                currentkey:0];

    return key;
}
@end

@implementation CKKSExternalTLKShare (CKKSTranslation)

- (instancetype)initWithViewName:(NSString*)viewName
                        tlkShare:(CKKSTLKShare*)tlkShareRecord
{
    return [self initWithView:viewName
                      tlkUUID:tlkShareRecord.tlkUUID
               receiverPeerID:[self datafyPeerID:tlkShareRecord.receiverPeerID]
                 senderPeerID:[self datafyPeerID:tlkShareRecord.senderPeerID]
                   wrappedTLK:tlkShareRecord.wrappedTLK
                    signature:tlkShareRecord.signature];
}


- (CKKSTLKShareRecord* _Nullable)makeTLKShareRecord:(CKRecordZoneID*)zoneID
                                          contextID:(NSString*)contextID
{
    CKKSTLKShare* tlkShare = [[CKKSTLKShare alloc] initForKey:self.tlkUUID
                                                 senderPeerID:[self stringifyPeerID:self.senderPeerID]
                                               recieverPeerID:[self stringifyPeerID:self.receiverPeerID]
                                     receiverEncPublicKeySPKI:nil
                                                        curve:SFEllipticCurveNistp384
                                                      version:SecCKKSTLKShareVersion1
                                                        epoch:0
                                                     poisoned:0
                                                   wrappedKey:self.wrappedTLK
                                                    signature:self.signature
                                                       zoneID:zoneID];

    return [[CKKSTLKShareRecord alloc] initWithShare:tlkShare
                                           contextID:contextID
                                              zoneID:zoneID
                                     encodedCKRecord:nil];
}

- (NSData*)datafyPeerID:(NSString*)peerID
{
    NSString* prefix = @"spid-";

    if ([peerID hasPrefix:prefix]) {
        peerID = [peerID substringFromIndex:[prefix length]];
    }

    return [[NSData alloc] initWithBase64EncodedString:peerID options:0];
}
@end

#endif
