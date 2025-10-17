/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
#include <string.h>
#include <sys/errno.h>
#include <stdint.h>

extern void   *secure_memset(void *, int, size_t);

/*
 * The memset_s function copies the value c into the first n bytes
 * pointed by s. No more than smax bytes will be copied.
 *
 * In contrast to the memset function, calls to memset_s will never
 * be ''optimised away'' by a compiler, ensuring the memory copy
 * even if s is not accessed anymore after this call.
 */
int
memset_s(void *s, size_t smax, int c, size_t n)
{
	int err = 0;

	if (s == NULL) {
		return EINVAL;
	}
	if (smax > RSIZE_MAX) {
		return E2BIG;
	}
	if (n > smax) {
		n = smax;
		err = EOVERFLOW;
	}

	/*
	 * secure_memset is defined in assembly, we therefore
	 * expect that the compiler will not inline the call.
	 */
	secure_memset(s, c, n);

	return err;
}

int
timingsafe_bcmp(const void *b1, const void *b2, size_t n)
{
	const unsigned char *p1 = b1, *p2 = b2;
	unsigned char ret = 0;

	for (; n > 0; n--) {
		ret |= *p1++ ^ *p2++;
	}

	/* map zero to zero and nonzero to one */
	return (ret + 0xff) >> 8;
}
