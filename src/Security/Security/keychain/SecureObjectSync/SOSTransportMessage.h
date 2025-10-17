/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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


#ifndef SOSTransportMessage_h
#define SOSTransportMessage_h

#import <Foundation/Foundation.h>
#include "keychain/SecureObjectSync/SOSAccountPriv.h"

@interface SOSMessage : NSObject

@property (nonatomic) SOSEngineRef engine;
@property (strong, nonatomic)  SOSAccount* account;
@property (strong, nonatomic)  NSString* circleName;

-(id) initWithAccount:(SOSAccount*)acct andName:(NSString*)name;

-(CFIndex) SOSTransportMessageGetTransportType;
-(CFStringRef) SOSTransportMessageGetCircleName;
-(CFTypeRef) SOSTransportMessageGetEngine;
-(SOSAccount*) SOSTransportMessageGetAccount;

-(bool) SOSTransportMessageCleanupAfterPeerMessages:(SOSMessage*) transport peers:(CFDictionaryRef) peers err:(CFErrorRef*) error;

-(bool) SOSTransportMessageSendMessage:(SOSMessage*) transport id:(CFStringRef) peerID messageToSend:(CFDataRef) message err:(CFErrorRef *)error;
-(bool) SOSTransportMessageSendMessages:(SOSMessage*) transport pm:(CFDictionaryRef) peer_messages err:(CFErrorRef *)error;
-(bool) SOSTransportMessageFlushChanges:(SOSMessage*) transport err:(CFErrorRef *)error;

-(bool) SOSTransportMessageSyncWithPeers:(SOSMessage*) transport p:(CFSetRef) peers err:(CFErrorRef *)error;

-(CFDictionaryRef)CF_RETURNS_RETAINED SOSTransportMessageHandlePeerMessageReturnsHandledCopy:(SOSMessage*) transport peerMessages:(CFMutableDictionaryRef) circle_peer_messages_table err:(CFErrorRef *)error;

-(bool) SOSTransportMessageHandlePeerMessage:(SOSMessage*) transport id:(CFStringRef) peer_id cm:(CFDataRef) codedMessage err:(CFErrorRef *)error;
-(bool) SOSTransportMessageSendMessageIfNeeded:(SOSMessage*) transport circleName:(CFStringRef) circle_id pID:(CFStringRef) peer_id err:(CFErrorRef *)error;
//for testing
bool SOSEngineHandleCodedMessage(SOSAccount* account, SOSEngineRef engine, CFStringRef peerID, CFDataRef codedMessage, CFErrorRef*error);
    
@end

#endif
