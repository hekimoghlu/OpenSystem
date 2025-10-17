/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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
#ifndef _SECOTRSESSIONPRIV_H_
#define _SECOTRSESSIONPRIV_H_

#include <CoreFoundation/CFBase.h>
#include <CoreFoundation/CFRuntime.h>
#include <CoreFoundation/CFDate.h>

#include <Security/SecOTR.h>
#include <corecrypto/ccn.h>
#include <corecrypto/ccmode.h>
#include <corecrypto/ccsha1.h>

#include <CommonCrypto/CommonDigest.h>

#include <dispatch/dispatch.h>

#include <Security/SecOTRMath.h>
#include <Security/SecOTRDHKey.h>
#include <Security/SecOTRSession.h>

__BEGIN_DECLS

typedef enum {
    kIdle,
    kAwaitingDHKey,
    kAwaitingRevealSignature,
    kAwaitingSignature,
    kDone
} SecOTRAuthState;

struct _SecOTRCacheElement {
    uint8_t _fullKeyHash[CCSHA1_OUTPUT_SIZE];
    uint8_t _publicKeyHash[CCSHA1_OUTPUT_SIZE];

    uint8_t _sendMacKey[kOTRMessageMacKeyBytes];
    uint8_t _sendEncryptionKey[kOTRMessageKeyBytes];

    uint8_t _receiveMacKey[kOTRMessageMacKeyBytes];
    uint8_t _receiveEncryptionKey[kOTRMessageKeyBytes];

    uint64_t _counter;
    uint64_t _theirCounter;
    
};
typedef struct _SecOTRCacheElement SecOTRCacheElement;

#define kOTRKeyCacheSize 4
#define kSecondsPerMinute 60

struct _SecOTRSession {
    CFRuntimeBase _base;
    
    SecOTRAuthState _state;
    
    SecOTRFullIdentityRef    _me;
    SecOTRPublicIdentityRef  _them;
    
    uint8_t _r[kOTRAuthKeyBytes];
    
    CFDataRef _receivedDHMessage;
    CFDataRef _receivedDHKeyMessage;

    uint32_t _keyID;
    SecOTRFullDHKeyRef _myKey;
    SecOTRFullDHKeyRef _myNextKey;

    uint32_t _theirKeyID;
    SecOTRPublicDHKeyRef _theirPreviousKey;
    SecOTRPublicDHKeyRef _theirKey;
    
    CFMutableDataRef _macKeysToExpose;

    dispatch_queue_t _queue;

    SecOTRCacheElement _keyCache[kOTRKeyCacheSize];
    
    bool _textOutput;
    bool _compactAppleMessages;
    bool _includeHashes;
    uint64_t _stallSeconds;

    bool _stallingTheirRoll;
    CFAbsoluteTime _timeToRoll;
    
    bool _missedAck;
    bool _receivedAck;
};

CFDataRef SecOTRCopyIncomingBytes(CFDataRef incomingMessage);
void SecOTRPrepareOutgoingBytes(CFMutableDataRef destinationMessage, CFMutableDataRef protectedMessage);

OSStatus SecOTRSetupInitialRemoteKey(SecOTRSessionRef session, SecOTRPublicDHKeyRef CF_CONSUMED initialKey);
void SOSOTRSRoll(SecOTRSessionRef session);
int SecOTRSGetKeyID(SecOTRSessionRef session);
int SecOTRSGetTheirKeyID(SecOTRSessionRef session);
void SecOTRSKickTimeToRoll(SecOTRSessionRef session);

__END_DECLS

#endif
