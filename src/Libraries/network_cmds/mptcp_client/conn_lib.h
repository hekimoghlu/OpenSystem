/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
//
//  Created by Anumita Biswas on 10/30/12.
//

#ifndef mptcp_client_conn_lib_h
#define mptcp_client_conn_lib_h

typedef struct conninfo {
	__uint32_t			ci_flags;			/* see flags in sys/socket.h (CIF_CONNECTING, etc...) */
	__uint32_t			ci_ifindex;			/* outbound interface */
	struct sockaddr		*ci_src;			/* source address */
	struct sockaddr		*ci_dst;			/* destination address */
	int					ci_error;			/* saved error */
	__uint32_t			ci_aux_type;		/* auxiliary data type */
	void				*ci_aux_data;		/* auxiliary data */
} conninfo_t;

extern int copyassocids(int, sae_associd_t **, uint32_t *);
extern void freeassocids(sae_associd_t *);
extern int copyconnids(int, sae_associd_t, sae_connid_t **, uint32_t *);
extern void freeconnids(sae_connid_t *);
extern int copyconninfo(int, sae_connid_t, conninfo_t **);
extern void freeconninfo(conninfo_t *);

#endif
