/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
#include <corecrypto/cc_debug.h>

void cc_print(const char *label, unsigned long count, const uint8_t *s) {
	size_t prefix_length = 0;

	if (label != NULL) {
		cc_printf("%s: ", label);
		prefix_length = strlen(label) + 2;
	} else {
		prefix_length = 0;
	}

	for (unsigned long index = 0; index < count; index++) {
		cc_printf("0x%02X, ", s[index]);

		if ((index % 16) == 0) {
			cc_printf("\n");

			if (prefix_length != 0) {
				for (size_t j = 0; j < prefix_length; j++) cc_printf(" ");
			}
		}
	}
}
