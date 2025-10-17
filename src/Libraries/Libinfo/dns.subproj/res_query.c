/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
#include "libinfo_common.h"

#include <dns_sd.h>
#include <errno.h>

#include <arpa/nameser_compat.h>
#include <nameser.h>

#include "si_module.h"

extern int h_errno;

// Storage for the global struct __res_9_state.
// The BIND9 libresolv.dylib shares the same storage for this structure as the
// legacy BIND8 libsystem_info.dylib. This implementation does not require the
// _res structure but libresolv.dylib does and many 3rd-party applications
// access this global symbol directly so we preserve it here.
#ifdef __LP64__
#define RES_9_STATE_SIZE 552
#else
#define RES_9_STATE_SIZE 512
#endif
LIBINFO_EXPORT
char _res[RES_9_STATE_SIZE] = {0};

LIBINFO_EXPORT
int
res_init(void)
{
	// For compatibility only.
	return 0;
}

// Perform a DNS query. Returned DNS response is placed in the answer buffer.
// A preliminary check of the answer is performed and success is returned only
// if no error is indicated in the answer and the answer count is nonzero.
// Returns the size of the response on success, or -1 with h_errno set.
static int
_mdns_query(int call, const char *name, int class, int type, u_char *answer, int anslen)
{
	int res = -1;
	si_item_t *item;
	uint32_t err;
	int cpylen = 0;
	
	si_mod_t *dns = si_module_with_name("mdns");
	if (dns == NULL) {
		h_errno = NO_RECOVERY;
		return -1;
	}
	
	item = dns->vtable->sim_item_call(dns, call, name, NULL, NULL, class, type, &err);
	
	if (item != NULL) {
		si_dnspacket_t *p;
		
		p = (si_dnspacket_t *)((uintptr_t)item + sizeof(si_item_t));
		
		res = p->dns_packet_len;
		
		if (res >= 0 && anslen >= 0) {
			// Truncate destination buffer size.
			memcpy(answer, p->dns_packet, (cpylen = MIN(res, anslen)));
		}
		else {
			h_errno = NO_RECOVERY;
			res = -1;
		}
		
		si_item_release(item);
	} else {
		h_errno = HOST_NOT_FOUND;
		res = -1;
	}

	if (cpylen >= sizeof(HEADER)) {
		HEADER *hp = (HEADER *)answer;
		switch (hp->rcode) {
			case NXDOMAIN:
				h_errno = HOST_NOT_FOUND;
				res = -1;
				break;
			case SERVFAIL:
				h_errno = TRY_AGAIN;
				res = -1;
				break;
			case NOERROR:
				if (ntohs(hp->ancount) == 0) {
					h_errno = NO_DATA;
					res = -1;
				}
				break;
			case FORMERR:
			case NOTIMP:
			case REFUSED:
			default:
				h_errno = NO_RECOVERY;
				res = -1;
				break;
		}
	}

	si_module_release(dns);
	return res;
}

LIBINFO_EXPORT
int
res_query(const char *name, int class, int type, u_char *answer, int anslen)
{
	return _mdns_query(SI_CALL_DNS_QUERY, name, class, type, answer, anslen);
}

LIBINFO_EXPORT
int
res_search(const char *name, int class, int type, u_char *answer, int anslen)
{
	return _mdns_query(SI_CALL_DNS_SEARCH, name, class, type, answer, anslen);
}
