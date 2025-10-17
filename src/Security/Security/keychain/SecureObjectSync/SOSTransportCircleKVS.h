/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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



#ifndef sec_SOSTransportCircleKVS_h
#define sec_SOSTransportCircleKVS_h

#import "SOSTransportCircle.h"
@class SOSCircleStorageTransport;

@interface SOSKVSCircleStorageTransport : SOSCircleStorageTransport
{
    NSMutableDictionary *pending_changes;
    NSString            *circleName;
}

@property (retain, nonatomic)   NSMutableDictionary *pending_changes;
@property (retain, nonatomic)   NSString            *circleName;


-(id)init;
-(id)initWithAccount:(SOSAccount*)acct andCircleName:(NSString*)name;
-(NSString*) getCircleName;
-(bool) flushChanges:(CFErrorRef *)error;

-(void)kvsAddToPendingChanges:(CFStringRef) message_key data:(CFDataRef)message_data;
-(bool)kvsSendPendingChanges:(CFErrorRef *)error;

-(bool)kvsAppendKeyInterest:(CFMutableArrayRef) alwaysKeys firstUnlock:(CFMutableArrayRef) afterFirstUnlockKeys unlocked:(CFMutableArrayRef)unlockedKeys err:(CFErrorRef *)error;
-(bool)kvsAppendRingKeyInterest:(CFMutableArrayRef) alwaysKeys firstUnlock:(CFMutableArrayRef)afterFirstUnlockKeys unlocked:(CFMutableArrayRef) unlockedKeys err:(CFErrorRef *)error;
-(bool)kvsAppendDebugKeyInterest:(CFMutableArrayRef) alwaysKeys firstUnlock:(CFMutableArrayRef)afterFirstUnlockKeys unlocked:(CFMutableArrayRef) unlockedKeys err:(CFErrorRef *)error;

-(bool) kvsRingFlushChanges:(CFErrorRef*) error;
-(bool) kvsRingPostRing:(CFStringRef) ringName ring:(CFDataRef) ring err:(CFErrorRef *)error;

-(bool) kvssendDebugInfo:(CFStringRef) type debug:(CFTypeRef) debugInfo  err:(CFErrorRef *)error;
-(bool) kvsSendOfficialDSID:(CFStringRef) dsid err:(CFErrorRef *)error;

@end;

#endif
