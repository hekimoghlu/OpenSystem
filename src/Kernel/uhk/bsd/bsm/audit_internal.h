/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
#ifndef _AUDIT_INTERNAL_H
#define _AUDIT_INTERNAL_H

#if defined(__linux__) && !defined(__unused)
#define __unused
#endif

#include <stddef.h>
#include <sys/queue.h>
#include <sys/types.h>

/*
 * audit_internal.h contains private interfaces that are shared by user space
 * and the kernel for the purposes of assembling audit records.  Applications
 * should not include this file or use the APIs found within, or it may be
 * broken with future releases of OpenBSM, which may delete, modify, or
 * otherwise break these interfaces or the assumptions they rely on.
 */
struct au_token {
	u_char                  *t_data;
	size_t                   len;
	TAILQ_ENTRY(au_token)    tokens;
};

struct au_record {
	char                     used;          /* Record currently in use? */
	int                      desc;          /* Descriptor for record. */
	TAILQ_HEAD(, au_token)   token_q;       /* Queue of BSM tokens. */
	u_char                  *data;
	size_t                   len;
	LIST_ENTRY(au_record)    au_rec_q;
};
typedef struct au_record        au_record_t;


/*
 * We could determined the header and trailer sizes by defining appropriate
 * structures.  We hold off that approach until we have a consistent way of
 * using structures for all tokens.  This is not straightforward since these
 * token structures may contain pointers of whose contents we do not know the
 * size (e.g text tokens).
 */
#define AUDIT_HEADER_EX_SIZE(a) ((a)->ai_termid.at_type+18+sizeof(u_int32_t))
#define AUDIT_HEADER_SIZE       18
#define MAX_AUDIT_HEADER_SIZE   (5*sizeof(u_int32_t)+18)
#define AUDIT_TRAILER_SIZE      7
#define MAX_AUDIT_IDENTITY_SIZE 179

/*
 * BSM token streams store fields in big endian byte order, so as to be
 * portable; when encoding and decoding, we must convert byte orders for
 * typed values.
 */
#define ADD_U_CHAR(loc, val)                                            \
	do {                                                            \
	        *(loc) = (val);                                         \
	        (loc) += sizeof(u_char);                                \
	} while(0)


#define ADD_U_INT16(loc, val)                                           \
	do {                                                            \
	        be16enc((loc), (val));                                  \
	        (loc) += sizeof(u_int16_t);                             \
	} while(0)

#define ADD_U_INT32(loc, val)                                           \
	do {                                                            \
	        be32enc((loc), (val));                                  \
	        (loc) += sizeof(u_int32_t);                             \
	} while(0)

#define ADD_U_INT64(loc, val)                                           \
	do {                                                            \
	        be64enc((loc), (val));                                  \
	        (loc) += sizeof(u_int64_t);                             \
	} while(0)

#define ADD_MEM(loc, data, size)                                        \
	do {                                                            \
	        memcpy((loc), (data), (size));                          \
	        (loc) += size;                                          \
	} while(0)

#define ADD_STRING(loc, data, size)     ADD_MEM(loc, data, size)

#endif /* !_AUDIT_INTERNAL_H_ */
