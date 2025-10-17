/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/kern_debug.h>

int
main(int argc, char *argv[])
{
	int opt;

	syscall_rejection_selector_t masks[16] = { 0 };

	int pos = 0;
	unsigned char selector = 0;
	bool next_is_allow = false;

	uint64_t flags = SYSCALL_REJECTION_FLAGS_DEFAULT;

	while ((opt = getopt(argc, argv, "ads:i:OF")) != -1) {
		switch (opt) {
		case 'a':
			next_is_allow = true;
			break;
		case 'd':
			next_is_allow = false;
			break;
		case 's':
			selector = (syscall_rejection_selector_t)atoi(optarg);
			break;
		case 'i':
			pos = atoi(optarg);
			if (next_is_allow) {
				// printf("%i: ALLOW %u\n", pos, (unsigned int)selector);
				masks[pos] = SYSCALL_REJECTION_ALLOW(selector);
			} else {
				// printf("%i: DENY %u\n", pos, (unsigned int)selector);
				masks[pos] = SYSCALL_REJECTION_DENY(selector);
			}
			break;
		case 'O':
			flags |= SYSCALL_REJECTION_FLAGS_ONCE;
			break;
		case 'F':
			flags |= SYSCALL_REJECTION_FLAGS_FORCE_FATAL;
			break;
		default:
			fprintf(stderr, "unknown option '%c'\n", opt);
			exit(2);
		}
	}

	debug_syscall_reject_config(masks, sizeof(masks) / sizeof(masks[0]), flags);

	int __unused ret = chdir("/tmp");

	syscall_rejection_selector_t all_allow_masks[16] = { 0 };
	all_allow_masks[0] = SYSCALL_REJECTION_ALLOW(SYSCALL_REJECTION_ALL);

	debug_syscall_reject_config(all_allow_masks, sizeof(all_allow_masks) / sizeof(all_allow_masks[0]), SYSCALL_REJECTION_FLAGS_DEFAULT);

	return 0;
}
