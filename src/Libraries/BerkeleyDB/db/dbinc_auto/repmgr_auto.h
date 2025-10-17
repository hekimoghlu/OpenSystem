/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
#ifndef	__repmgr_AUTO_H
#define	__repmgr_AUTO_H

/*
 * Message sizes are simply the sum of field sizes (not
 * counting variable size parts, when DBTs are present),
 * and may be different from struct sizes due to padding.
 */
#define	__REPMGR_HANDSHAKE_SIZE	6
typedef struct ___repmgr_handshake_args {
	u_int16_t	port;
	u_int32_t	priority;
} __repmgr_handshake_args;

#define	__REPMGR_ACK_SIZE	12
typedef struct ___repmgr_ack_args {
	u_int32_t	generation;
	DB_LSN		lsn;
} __repmgr_ack_args;

#define	__REPMGR_VERSION_PROPOSAL_SIZE	8
typedef struct ___repmgr_version_proposal_args {
	u_int32_t	min;
	u_int32_t	max;
} __repmgr_version_proposal_args;

#define	__REPMGR_VERSION_CONFIRMATION_SIZE	4
typedef struct ___repmgr_version_confirmation_args {
	u_int32_t	version;
} __repmgr_version_confirmation_args;

#define	__REPMGR_MAXMSG_SIZE	12
#endif
