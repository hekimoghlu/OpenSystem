/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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
#ifndef ND_CPACK_H
#define ND_CPACK_H

#include "netdissect.h"

struct cpack_state {
	const uint8_t					*c_buf;
	const uint8_t					*c_next;
	size_t						 c_len;
};

int nd_cpack_init(struct cpack_state *, const uint8_t *, size_t);

int nd_cpack_uint8(netdissect_options *, struct cpack_state *, uint8_t *);
int nd_cpack_int8(netdissect_options *, struct cpack_state *, int8_t *);
int nd_cpack_uint16(netdissect_options *, struct cpack_state *, uint16_t *);
int nd_cpack_int16(netdissect_options *, struct cpack_state *, int16_t *);
int nd_cpack_uint32(netdissect_options *, struct cpack_state *, uint32_t *);
int nd_cpack_int32(netdissect_options *, struct cpack_state *, int32_t *);
int nd_cpack_uint64(netdissect_options *, struct cpack_state *, uint64_t *);
int nd_cpack_int64(netdissect_options *, struct cpack_state *, int64_t *);

const uint8_t *nd_cpack_next_boundary(const uint8_t *buf, const uint8_t *p, size_t alignment);
const uint8_t *nd_cpack_align_and_reserve(struct cpack_state *cs, size_t wordsize);

extern int nd_cpack_advance(struct cpack_state *, const size_t);

#endif /* ND_CPACK_H */
