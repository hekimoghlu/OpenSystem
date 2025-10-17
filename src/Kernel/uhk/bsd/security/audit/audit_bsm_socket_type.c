/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#include <sys/param.h>
#include <sys/socket.h>

#include <security/audit/audit.h>

#include <bsm/audit_record.h>
#include <bsm/audit_socket_type.h>

#if CONFIG_AUDIT
struct bsm_socket_type {
	u_short bst_bsm_socket_type;
	int     bst_local_socket_type;
};

#define ST_NO_LOCAL_MAPPING     -600

static const struct bsm_socket_type bsm_socket_types[] = {
	{ BSM_SOCK_DGRAM, SOCK_DGRAM },
	{ BSM_SOCK_STREAM, SOCK_STREAM },
	{ BSM_SOCK_RAW, SOCK_RAW },
	{ BSM_SOCK_RDM, SOCK_RDM },
	{ BSM_SOCK_SEQPACKET, SOCK_SEQPACKET },
};
static const int bsm_socket_types_count = sizeof(bsm_socket_types) /
    sizeof(bsm_socket_types[0]);

static const struct bsm_socket_type *
bsm_lookup_local_socket_type(int local_socket_type)
{
	int i;

	for (i = 0; i < bsm_socket_types_count; i++) {
		if (bsm_socket_types[i].bst_local_socket_type ==
		    local_socket_type) {
			return &bsm_socket_types[i];
		}
	}
	return NULL;
}

u_short
au_socket_type_to_bsm(int local_socket_type)
{
	const struct bsm_socket_type *bstp;

	bstp = bsm_lookup_local_socket_type(local_socket_type);
	if (bstp == NULL) {
		return BSM_SOCK_UNKNOWN;
	}
	return bstp->bst_bsm_socket_type;
}

static const struct bsm_socket_type *
bsm_lookup_bsm_socket_type(u_short bsm_socket_type)
{
	int i;

	for (i = 0; i < bsm_socket_types_count; i++) {
		if (bsm_socket_types[i].bst_bsm_socket_type ==
		    bsm_socket_type) {
			return &bsm_socket_types[i];
		}
	}
	return NULL;
}

int
au_bsm_to_socket_type(u_short bsm_socket_type, int *local_socket_typep)
{
	const struct bsm_socket_type *bstp;

	bstp = bsm_lookup_bsm_socket_type(bsm_socket_type);
	if (bstp == NULL || bstp->bst_local_socket_type) {
		return -1;
	}
	*local_socket_typep = bstp->bst_local_socket_type;
	return 0;
}
#endif /* CONFIG_AUDIT */
