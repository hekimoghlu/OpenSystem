/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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



#ifndef SOSTransport_h
#define SOSTransport_h
#include "keychain/SecureObjectSync/SOSTransportMessage.h"
#include "keychain/SecureObjectSync/SOSTransportCircle.h"
#include "keychain/SecureObjectSync/SOSTransportKeyParameter.h"
#include "keychain/SecureObjectSync/SOSAccount.h"

CF_RETURNS_RETAINED CFMutableArrayRef SOSTransportDispatchMessages(SOSAccountTransaction* txn, CFDictionaryRef updates, CFErrorRef *error);

void SOSRegisterTransportMessage(SOSMessage* additional);
void SOSUnregisterTransportMessage(SOSMessage* removal);

void SOSRegisterTransportCircle(SOSCircleStorageTransport* additional);
void SOSUnregisterTransportCircle(SOSCircleStorageTransport* removal);

void SOSRegisterTransportKeyParameter(CKKeyParameter* additional);
void SOSUnregisterTransportKeyParameter(CKKeyParameter* removal);
void SOSUnregisterAllTransportMessages(void);
void SOSUnregisterAllTransportCircles(void);
void SOSUnregisterAllTransportKeyParameters(void);


void SOSUpdateKeyInterest(SOSAccount* account);

enum TransportType{
    kUnknown = 0,
    kKVS = 1,
    kIDS = 2,
    kBackupPeer = 3,
    kIDSTest = 4,
    kKVSTest = 5,
    kCK = 6
};

static inline CFMutableDictionaryRef CFDictionaryEnsureCFDictionaryAndGetCurrentValue(CFMutableDictionaryRef dict, CFTypeRef key)
{
    CFMutableDictionaryRef result = (CFMutableDictionaryRef) CFDictionaryGetValue(dict, key);

    if (!isDictionary(result)) {
        result = CFDictionaryCreateMutableForCFTypes(kCFAllocatorDefault);
        CFDictionarySetValue(dict, key, result);
        CFReleaseSafe(result);
    }

    return result;
}

#endif
