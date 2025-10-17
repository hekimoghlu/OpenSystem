/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
#ifndef _EAP8021X_EAPTLS_H
#define _EAP8021X_EAPTLS_H

/*
 * EAPTLS.h
 * - definitions for EAP-TLS
 */

/* 
 * Modification History
 *
 * August 26, 2002	Dieter Siegmund (dieter@apple)
 * - created
 */

#include <stdint.h>

typedef struct {
    uint8_t		code;
    uint8_t		identifier;
    uint8_t		length[2];	/* of entire request/response */
    uint8_t		type;
    uint8_t		flags;
    uint8_t		tls_data[0];
} EAPTLSPacket, *EAPTLSPacketRef, EAPTLSFragment, *EAPTLSFragmentRef;

typedef struct {
    uint8_t		code;
    uint8_t		identifier;
    uint8_t		length[2];	/* of entire request/response */
    uint8_t		type;
    uint8_t		flags;
    uint8_t		tls_message_length[4]; /* if flags.L == 1 */
    uint8_t		tls_data[0];
} EAPTLSLengthIncludedPacket, *EAPTLSLengthIncludedPacketRef;

typedef enum {
    kEAPTLSPacketFlagsLengthIncluded	= 0x80,
    kEAPTLSPacketFlagsMoreFragments 	= 0x40,
    kEAPTLSPacketFlagsStart 		= 0x20,
} EAPTLSPacketFlags;

uint32_t
EAPTLSLengthIncludedPacketGetMessageLength(EAPTLSLengthIncludedPacketRef pkt);

void
EAPTLSLengthIncludedPacketSetMessageLength(EAPTLSLengthIncludedPacketRef pkt, 
					   uint32_t length);
#endif /* _EAP8021X_EAPTLS_H */
