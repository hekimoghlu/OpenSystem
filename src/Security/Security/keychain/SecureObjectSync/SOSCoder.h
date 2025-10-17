/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
#ifndef _SEC_SOSCODER_H_
#define _SEC_SOSCODER_H_


#include "keychain/SecureObjectSync/SOSFullPeerInfo.h"
#include <Security/SecureObjectSync/SOSPeerInfo.h>

typedef struct __OpaqueSOSCoder *SOSCoderRef;
typedef bool (^SOSPeerSendBlock)(CFDataRef message, CFErrorRef *error);

enum {
    kSOSCoderDataReturned = 0,
    kSOSCoderNegotiating = 1,
    kSOSCoderNegotiationCompleted = 2,
    kSOSCoderFailure = 3,
    kSOSCoderStaleEvent = 4,
    kSOSCoderTooNew = 5,
    kSOSCoderForceMessage = 6,
};

typedef uint32_t SOSCoderStatus;

CFTypeID SOSCoderGetTypeID(void);

SOSCoderRef SOSCoderCreate(SOSPeerInfoRef peerInfo, SOSFullPeerInfoRef myPeerInfo, CFBooleanRef useCompact, CFErrorRef *error);
SOSCoderRef SOSCoderCreateFromData(CFDataRef exportedData, CFErrorRef *error);

CFDataRef SOSCoderCopyDER(SOSCoderRef coder, CFErrorRef* error);

CFStringRef SOSCoderGetID(SOSCoderRef coder);

bool SOSCoderIsFor(SOSCoderRef coder, SOSPeerInfoRef peerInfo, SOSFullPeerInfoRef myPeerInfo);

SOSCoderStatus
SOSCoderStart(SOSCoderRef coder, CFErrorRef *error);

SOSCoderStatus
SOSCoderResendDH(SOSCoderRef coder, CFErrorRef *error);

void SOSCoderPersistState(CFStringRef peer_id, SOSCoderRef coder);

SOSCoderStatus SOSCoderUnwrap(SOSCoderRef coder, CFDataRef codedMessage, CFMutableDataRef *message,
                              CFStringRef clientId, CFErrorRef *error);

SOSCoderStatus SOSCoderWrap(SOSCoderRef coder, CFDataRef message, CFMutableDataRef *codedMessage, CFStringRef clientId, CFErrorRef *error);

bool SOSCoderCanWrap(SOSCoderRef coder);

void SOSCoderReset(SOSCoderRef coder);

CFDataRef SOSCoderCopyPendingResponse(SOSCoderRef coder);
void SOSCoderConsumeResponse(SOSCoderRef coder);
bool SOSCoderIsCoderInAwaitingState(SOSCoderRef coder);
    
#endif // _SEC_SOSCODER_H_
