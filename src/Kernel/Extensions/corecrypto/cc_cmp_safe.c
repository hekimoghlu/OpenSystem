/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#include <stddef.h>
#include <corecrypto/cc.h>

int cc_cmp_safe(size_t num, const void *ptr1, const void *ptr2) {
	if (num == 0) return 1;

	volatile const unsigned char *buffer1 = ptr1;
	volatile const unsigned char *buffer2 = ptr2;
	unsigned char result = 0;

	while (num != 0) {
		result |= *buffer1 ^ *buffer2;
		buffer1++;
		buffer2++;
		num--;
	}

	return !!result;
}
