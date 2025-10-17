/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#ifdef BUILDING_SIMPLE
#include <stdint.h>
#else
#ifdef HAVE_STDINT_H
#include <stdint.h>
#else
#include <uuid/uuid_types.h>
#endif
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#endif /* BUILDING_SIMPLE */

#include <uuid/uuid.h>

#ifdef BUILDING_SIMPLE
#define _UUID_INTERNAL __attribute__((visibility("hidden")))
#ifndef UUID_UNPARSE_DEFAULT_UPPER
#define UUID_UNPARSE_DEFAULT_UPPER
#endif
#else
#define _UUID_INTERNAL
#endif /* BUILDING_SIMPLE */

/*
 * Offset between 15-Oct-1582 and 1-Jan-70
 */
#define TIME_OFFSET_HIGH 0x01B21DD2
#define TIME_OFFSET_LOW  0x13814000

struct uuid {
	uint32_t	time_low;
	uint16_t	time_mid;
	uint16_t	time_hi_and_version;
	uint16_t	clock_seq;
	uint8_t	node[6];
};

/* UUID Variant definitions */
#define UUID_VARIANT_NCS 	0
#define UUID_VARIANT_DCE 	1
#define UUID_VARIANT_MICROSOFT	2
#define UUID_VARIANT_OTHER	3

/* UUID Type definitions */
#define UUID_TYPE_DCE_TIME   1
#define UUID_TYPE_DCE_RANDOM 4

/*
 * prototypes
 */
void uuid_pack(const struct uuid *uu, uuid_t ptr) _UUID_INTERNAL;
void uuid_unpack(const uuid_t in, struct uuid *uu) _UUID_INTERNAL;

#if !defined(BUILDING_SIMPLE)
time_t uuid_time(const uuid_t uu, struct timeval *ret_tv);
#endif /* !defined(BUILDING_SIMPLE) */
int uuid_type(const uuid_t uu);
int uuid_variant(const uuid_t uu);
