/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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


#ifndef SOSTransportCircle_h
#define SOSTransportCircle_h
#import "keychain/SecureObjectSync/SOSPeerInfo.h"

@class SOSAccount;

@interface SOSCircleStorageTransport : NSObject
{
    SOSAccount* account;
}

@property (retain, nonatomic) SOSAccount* account;

-(id) init;
-(SOSCircleStorageTransport*) initWithAccount:(SOSAccount*)account;

-(CFIndex)circleGetTypeID;
-(CFIndex)getTransportType;
-(SOSAccount*)getAccount;

-(bool) expireRetirementRecords:(CFDictionaryRef) retirements err:(CFErrorRef *)error;

-(bool) flushChanges:(CFErrorRef *)error;
-(bool) postCircle:(CFStringRef)circleName circleData:(CFDataRef)circle_data err:(CFErrorRef *)error;
-(bool) postRetirement:(CFStringRef) circleName peer:(SOSPeerInfoRef) peer err:(CFErrorRef *)error;

-(CFDictionaryRef) CF_RETURNS_RETAINED handleRetirementMessages:(CFMutableDictionaryRef) circle_retirement_messages_table err:(CFErrorRef *)error;
-(CFArrayRef)CF_RETURNS_RETAINED handleCircleMessagesAndReturnHandledCopy:(CFMutableDictionaryRef) circle_circle_messages_table err:(CFErrorRef *)error;

@end
#endif
