/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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
#ifndef _SECOTRSESSION_H_
#define _SECOTRSESSION_H_

#include <CoreFoundation/CFBase.h>
#include <CoreFoundation/CFData.h>

#include <Security/SecOTR.h>

__BEGIN_DECLS

// MARK: MessageTypes

enum SecOTRSMessageKind {
    kOTRNegotiationPacket,
    kOTRDataPacket,
    kOTRUnknownPacket
};

// MARK: OTR Session

enum SecOTRCreateFlags {
    kSecOTRSendTextMessages = 1 << 0, // OTR messages will be encoded as Base-64 with header/footer per the standard, not just given back in binary
    kSecOTRUseAppleCustomMessageFormat = 1 << 1, // OTR Messages will be encoded without revealing MAC keys and as compact as we can (P-256)
    kSecOTRIncludeHashesInMessages = 1 << 2,
    kSecOTRSlowRoll = 1 << 3,
};

/*!
 @typedef
 @abstract   OTRSessions encapsulate a commuincaiton between to parties using the
             otr protocol.
 @discussion Sessions start with IDs. One end sends a start packet (created with AppendStartPacket).
             Both sides process packets they exchange on the negotiation channel.
 */
typedef struct _SecOTRSession* SecOTRSessionRef;

SecOTRSessionRef SecOTRSessionCreateFromID(CFAllocatorRef allocator,
                                           SecOTRFullIdentityRef myID,
                                           SecOTRPublicIdentityRef theirID);

SecOTRSessionRef SecOTRSessionCreateFromIDAndFlags(CFAllocatorRef allocator,
                                           SecOTRFullIdentityRef myID,
                                           SecOTRPublicIdentityRef theirID,
                                           uint32_t flags);

SecOTRSessionRef SecOTRSessionCreateFromData(CFAllocatorRef allocator, CFDataRef data);

    void SecOTRSessionReset(SecOTRSessionRef session);
OSStatus SecOTRSAppendSerialization(SecOTRSessionRef publicID, CFMutableDataRef serializeInto);

OSStatus SecOTRSAppendStartPacket(SecOTRSessionRef session, CFMutableDataRef appendInitiatePacket);

OSStatus SecOTRSAppendRestartPacket(SecOTRSessionRef session, CFMutableDataRef appendPacket);

OSStatus SecOTRSProcessPacket(SecOTRSessionRef session,
                              CFDataRef incomingPacket,
                              CFMutableDataRef negotiationResponse);
    
bool SecOTRSIsForKeys(SecOTRSessionRef session, SecKeyRef myPublic, SecKeyRef theirPublic);
bool SecOTRSGetIsReadyForMessages(SecOTRSessionRef session);
bool SecOTRSGetIsIdle(SecOTRSessionRef session);

enum SecOTRSMessageKind SecOTRSGetMessageKind(SecOTRSessionRef session, CFDataRef incomingPacket);

/*!
 @function
 @abstract   Precalculates keys for current key sets to save time when sending or receiving.
 @param      session                OTRSession receiving message
 */
void SecOTRSPrecalculateKeys(SecOTRSessionRef session);
    
/*!
 @function
 @abstract   Encrypts and Signs a message with OTR credentials.
 @param      session                OTRSession receiving message
 @param      sourceMessage          Cleartext message to protect
 @param      protectedMessage       Data to append the encoded protected message to
 @result     OSStatus               errSecAuthFailed -> bad signature, no data appended.
 */

OSStatus SecOTRSSignAndProtectMessage(SecOTRSessionRef session,
                                      CFDataRef sourceMessage,
                                      CFMutableDataRef protectedMessage);

/*!
 @function
 @abstract   Verifies and exposes a message sent via OTR
 @param      session                OTRSession receiving message
 @param      incomingMessage        Encoded message
 @param      exposedMessageContents Data to append the exposed message to
 @result     OSStatus               errSecAuthFailed -> bad signature, no data appended.
 */

OSStatus SecOTRSVerifyAndExposeMessage(SecOTRSessionRef session,
                                       CFDataRef incomingMessage,
                                       CFMutableDataRef exposedMessageContents);



const char *SecOTRPacketTypeString(CFDataRef message);

CFDataRef SecOTRSessionCreateRemote(CFDataRef publicPeerId, CFErrorRef *error);
bool SecOTRSessionProcessPacketRemote(CFDataRef sessionData, CFDataRef inputPacket, CFDataRef* outputSessionData, CFDataRef* outputPacket, bool *readyForMessages, CFErrorRef *error);

bool SecOTRSessionIsSessionInAwaitingState(SecOTRSessionRef session);

__END_DECLS

#endif
