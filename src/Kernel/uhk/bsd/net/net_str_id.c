/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#include <sys/types.h>
#include <kern/locks.h>
#include <kern/zalloc.h>
#include <sys/errno.h>
#include <sys/sysctl.h>
#include <sys/malloc.h>
#include <sys/socket.h>
#include <libkern/OSAtomic.h>
#include <libkern/libkern.h>
#include <net/if.h>
#include <net/if_mib.h>
#include <string.h>

// TODO: -fbounds-safety increases the alignment and we have
//       no control over the alignment. (rdar://118519573)
#pragma clang diagnostic ignored "-Wcast-align"
#include "net/net_str_id.h"

#define NET_ID_STR_MAX_LEN 2048

#define FIRST_NET_STR_ID                                1000
static SLIST_HEAD(, net_str_id_entry)    net_str_id_list = {NULL};
static LCK_GRP_DECLARE(net_str_id_grp, "mbuf_tag_allocate_id");
static LCK_MTX_DECLARE(net_str_id_lock, &net_str_id_grp);

static u_int32_t nsi_kind_next[NSI_MAX_KIND] = { FIRST_NET_STR_ID, FIRST_NET_STR_ID, FIRST_NET_STR_ID };
static u_int32_t nsi_next_id = FIRST_NET_STR_ID;

__private_extern__ void
net_str_id_first_last(u_int32_t *first, u_int32_t *last, u_int32_t kind)
{
	*first = FIRST_NET_STR_ID;

	switch (kind) {
	case NSI_MBUF_TAG:
	case NSI_VENDOR_CODE:
	case NSI_IF_FAM_ID:
		*last = nsi_kind_next[kind] - 1;
		break;
	default:
		*last = FIRST_NET_STR_ID - 1;
		break;
	}
}

__private_extern__ errno_t
net_str_id_find_internal(const char *string, u_int32_t *out_id,
    u_int32_t kind, int create)
{
	struct net_str_id_entry                 *entry = NULL;


	if (string == NULL || out_id == NULL || kind >= NSI_MAX_KIND) {
		return EINVAL;
	}
	if (strlen(string) > NET_ID_STR_MAX_LEN) {
		return EINVAL;
	}

	*out_id = 0;

	/* Look for an existing entry */
	lck_mtx_lock(&net_str_id_lock);
	SLIST_FOREACH(entry, &net_str_id_list, nsi_next) {
		if (strlcmp(entry->nsi_string, string, entry->nsi_length) == 0) {
			break;
		}
	}

	if (entry == NULL) {
		if (create == 0) {
			lck_mtx_unlock(&net_str_id_lock);
			return ENOENT;
		}

		const uint32_t string_length = (uint32_t)strlen(string) + 1;
		entry = zalloc_permanent(sizeof(*entry) + string_length,
		    ZALIGN_PTR);
		if (entry == NULL) {
			lck_mtx_unlock(&net_str_id_lock);
			return ENOMEM;
		}

		strlcpy(entry->nsi_string, string, string_length);
		entry->nsi_length = string_length;
		entry->nsi_flags = (1 << kind);
		entry->nsi_id = nsi_next_id++;
		nsi_kind_next[kind] = nsi_next_id;
		SLIST_INSERT_HEAD(&net_str_id_list, entry, nsi_next);
	} else if ((entry->nsi_flags & (1 << kind)) == 0) {
		if (create == 0) {
			lck_mtx_unlock(&net_str_id_lock);
			return ENOENT;
		}
		entry->nsi_flags |= (1 << kind);
		if (entry->nsi_id >= nsi_kind_next[kind]) {
			nsi_kind_next[kind] = entry->nsi_id + 1;
		}
	}
	lck_mtx_unlock(&net_str_id_lock);

	*out_id = entry->nsi_id;

	return 0;
}
