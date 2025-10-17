/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
#ifndef _EAP8021X_EAP_H
#define _EAP8021X_EAP_H


/* 
 * Modification History
 *
 * November 1, 2001	Dieter Siegmund (dieter@apple.com)
 * - created
 */

/*
 * EAP.h
 * - EAP protocol definitions
 */

#include <stdint.h>
#include <sys/types.h>

enum {
    kEAPCodeRequest = 1,
    kEAPCodeResponse = 2,
    kEAPCodeSuccess = 3,
    kEAPCodeFailure = 4,
};
typedef uint32_t EAPCode;

enum {
    kEAPTypeInvalid = 0,		/* 0 is invalid */
    kEAPTypeIdentity = 1,
    kEAPTypeNotification = 2,
    kEAPTypeNak = 3,
    kEAPTypeMD5Challenge = 4,
    kEAPTypeOneTimePassword = 5,
    kEAPTypeGenericTokenCard = 6,
    kEAPTypeTLS = 13,
    kEAPTypeCiscoLEAP = 17,
    kEAPTypeEAPSIM = 18,
    kEAPTypeSRPSHA1 = 19,
    kEAPTypeTTLS = 21,
    kEAPTypeEAPAKA = 23,
    kEAPTypePEAP = 25,
    kEAPTypeMSCHAPv2 = 26,
    kEAPTypeExtensions = 33,
    kEAPTypeEAPFAST = 43,
    kEAPTypeEAPAKAPrime = 50,
};
typedef uint32_t EAPType;

typedef struct EAPPacket_s {
    uint8_t		code;
    uint8_t		identifier;
    uint8_t		length[2];		/* of entire request/response */
    uint8_t		data[0];
} EAPPacket, *EAPPacketRef;

typedef struct EAPSuccessFailurePacket_s {
    uint8_t		code;
    uint8_t		identifier;
    uint8_t		length[2];
} EAPSuccessPacket, *EAPSuccessPacketRef, 
    EAPFailurePacket, *EAPFailurePacketRef;

typedef struct EAPRequestResponsePacket_s {
    uint8_t		code;
    uint8_t		identifier;
    uint8_t		length[2];	/* of entire request/response */
    uint8_t		type;		/* EAPType values */
    uint8_t		type_data[0];
} EAPRequestPacket, *EAPRequestPacketRef, 
    EAPResponsePacket, *EAPResponsePacketRef,
    EAPNakPacket, *EAPNakPacketRef;

typedef struct EAPNotificationPacket_s {
    uint8_t		code;
    uint8_t		identifier;
    uint8_t		length[2];	/* sizeof(EAPNotificationPacket) */
    uint8_t		type;		/* kEAPTypeNotification */
} EAPNotificationPacket, *EAPNotificationPacketRef;

typedef struct EAPMD5ChallengePacket_s {
    uint8_t		code;
    uint8_t		identifier;
    uint8_t		length[2];	/* of entire request/response */
    uint8_t		type;
    uint8_t		value_size;
    uint8_t		value[0];
    /*
      uint8_t		name[0];
    */
} EAPMD5ChallengePacket, *EAPMD5ChallengePacketRef;

typedef struct EAPMD5ResponsePacket_s {
    uint8_t		code;
    uint8_t		identifier;
    uint8_t		length[2];	/* of entire request/response */
    uint8_t		type;
    uint8_t		value_size;	/* will be 16 */
    uint8_t		value[16];
    uint8_t		name[0];
} EAPMD5ResponsePacket, *EAPMD5ResponsePacketRef;

void
EAPPacketSetLength(EAPPacketRef pkt, uint16_t length);

uint16_t
EAPPacketGetLength(const EAPPacketRef pkt);

#define kEAPMasterSessionKeyMinimumSize			64
#define kEAPExtendedMasterSessionKeyMinimumSize		64

#endif /* _EAP8021X_EAP_H */
