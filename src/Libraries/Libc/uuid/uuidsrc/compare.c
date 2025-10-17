/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
#include "uuidP.h"
#include <string.h>

#define UUCMP(u1,u2) if (u1 != u2) return((u1 < u2) ? -1 : 1);

int uuid_compare(const uuid_t uu1, const uuid_t uu2)
{
	struct uuid	uuid1, uuid2;

	uuid_unpack(uu1, &uuid1);
	uuid_unpack(uu2, &uuid2);

	UUCMP(uuid1.time_low, uuid2.time_low);
	UUCMP(uuid1.time_mid, uuid2.time_mid);
	UUCMP(uuid1.time_hi_and_version, uuid2.time_hi_and_version);
	UUCMP(uuid1.clock_seq, uuid2.clock_seq);
	return memcmp(uuid1.node, uuid2.node, 6);
}

