/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
#ifndef _EAP8021X_EAPUTIL_H
#define _EAP8021X_EAPUTIL_H


/*
 * EAPUtil.h
 * - functions to return string values, and validate values for EAP
 */

#include <EAP8021X/EAP.h>
#include <stdbool.h>
#include <stdio.h>
#include <CoreFoundation/CFString.h>

int
EAPCodeValid(EAPCode code);

const char *
EAPCodeStr(EAPCode code);

const char *
EAPTypeStr(EAPType type);

bool
EAPPacketValid(EAPPacketRef eap_p, uint16_t pkt_length, FILE * f);

bool
EAPPacketIsValid(EAPPacketRef eap_p, uint16_t pkt_length,
		 CFMutableStringRef str);

/*
 * Function: EAPPacketCreate
 *
 * Purpose:
 *   Create an EAP packet, filling in the header information, and optionally,
 *   the type and its associated data.
 *
 *   If type is kEAPTypeInvalid, the packet size will be sizeof(EAPPacket),
 *   data and data_len are ignored.
 *
 *   If type is not kEAPTypeInvalid, the packet size will be 
 *   (sizeof(EAPRequestPacket) + data_len).  If data is not NULL, data_len
 *   bytes are copied into the type_data field.  If data is NULL, the caller
 *   fills in the data.
 *   
 *   If buf is not NULL, use it if it's big enough, otherwise
 *   malloc() a buffer that's large enough.
 *
 * Returns:
 *   A pointer to buf, if the buffer was used, otherwise a newly allocated
 *   buffer that must be released by calling free(), and also, the total
 *   length of the packet, in ret_size_p.
 *
 * Code example:
 *
 *   char 		buf[20];
 *   int 		identifier = 123;
 *   const char * 	identity = "user@domain";
 *   EAPPacketRef 	pkt;
 *   int		size;
 *
 *   pkt = EAPPacketCreate(buf, sizeof(buf), kEAPCodeResponse,
 *                         identifier, kEAPTypeIdentity, identity,
 *			   sizeof(identity), &size);
 *   send_packet(pkt);
 *   if (pkt != buf) {
 *       free(pkt);
 *   }
 */
EAPPacketRef
EAPPacketCreate(void * buf, int buf_size, 
		uint8_t code, int identifier, int type,
		const void * data, int data_len,
		int * ret_size_p);

#endif /* _EAP8021X_EAPUTIL_H */

