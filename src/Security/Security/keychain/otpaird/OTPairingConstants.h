/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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

#define OTPairingMachServiceName    "com.apple.security.otpaird"

#define OTPairingIDSServiceName     @"com.apple.private.alloy.octagon"

#define OTPairingIDSKeyMessageType  @"m"
#define OTPairingIDSKeySession      @"session"
#define OTPairingIDSKeyPacket       @"packet"
#define OTPairingIDSKeyErrorDescription @"error"

enum OTPairingIDSMessageType {
    OTPairingIDSMessageTypePacket = 1,
    OTPairingIDSMessageTypeError =  2,
    OTPairingIDSMessageTypePoke =   3,
};

#define OTPairingXPCKeyOperation    "operation"
#define OTPairingXPCKeyImmediate    "immediate"
#define OTPairingXPCKeyError        "error"
#define OTPairingXPCKeySuccess      "success"

#define OTPairingErrorDomain        @"com.apple.security.otpaird"
enum OTPairingErrorType {
    OTPairingSuccess = 0,
    OTPairingErrorTypeLock = 1,
    OTPairingErrorTypeXPC = 2,
    OTPairingErrorTypeIDS = 3,
    OTPairingErrorTypeRemote = 4,
    OTPairingErrorTypeAlreadyIn = 5,
    OTPairingErrorTypeBusy = 6,
    OTPairingErrorTypeKCPairing = 7,
    OTPairingErrorTypeSessionTimeout = 8,
    OTPairingErrorTypeRetryScheduled = 9,
};

enum {
    OTPairingOperationInitiate = 1,
};

#define OTPairingXPCActivityIdentifier  "com.apple.security.otpaird.pairing"
#define OTPairingXPCActivityInterval    XPC_ACTIVITY_INTERVAL_15_MIN

#define OTPairingXPCActivityPoke        "com.apple.security.otpaird.poke"

#define OTPairingXPCEventIDSDeviceState "ids-device-state"
