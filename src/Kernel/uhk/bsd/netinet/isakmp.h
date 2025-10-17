/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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
/*
 * RFC 2408 Internet Security Association and Key Management Protocol
 */

#ifndef _NETINET_ISAKMP_H_
#define _NETINET_ISAKMP_H_

typedef u_char cookie_t[8];
typedef u_char msgid_t[4];

/* 3.1 ISAKMP Header Format (IKEv1 and IKEv2)
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  !                          Initiator                            !
 *  !                            Cookie                             !
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  !                          Responder                            !
 *  !                            Cookie                             !
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  !  Next Payload ! MjVer ! MnVer ! Exchange Type !     Flags     !
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  !                          Message ID                           !
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  !                            Length                             !
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */
struct isakmp {
	cookie_t i_ck;          /* Initiator Cookie */
	cookie_t r_ck;          /* Responder Cookie */
	uint8_t np;             /* Next Payload Type */
	uint8_t vers;
#define ISAKMP_VERS_MAJOR       0xf0
#define ISAKMP_VERS_MAJOR_SHIFT 4
#define ISAKMP_VERS_MINOR       0x0f
#define ISAKMP_VERS_MINOR_SHIFT 0
	uint8_t etype;          /* Exchange Type */
	uint8_t flags;          /* Flags */
	msgid_t msgid;
	uint32_t len;           /* Length */
};

/* 3.2 Payload Generic Header
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  ! Next Payload  !   RESERVED    !         Payload Length        !
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */
struct isakmp_gen {
	uint8_t  np;       /* Next Payload */
	uint8_t  critical; /* bit 7 - critical, rest is RESERVED */
	uint16_t len;      /* Payload Length */
};

#endif /* _NETINET_ISAKMP_H_ */
