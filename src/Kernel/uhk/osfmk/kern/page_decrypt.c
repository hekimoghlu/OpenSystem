/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
#include <debug.h>

#include <kern/page_decrypt.h>
#include <kern/task.h>
#include <machine/commpage.h>

static dsmos_page_transform_hook_t dsmos_hook;

void
dsmos_page_transform_hook(dsmos_page_transform_hook_t hook)
{
	printf("DSMOS has arrived\n");
	/* set the hook now - new callers will run with it */
	dsmos_hook = hook;
}

int
dsmos_page_transform(const void* from, void *to, unsigned long long src_offset, void *ops)
{
	static boolean_t first_wait = TRUE;

	if (dsmos_hook == NULL) {
		if (first_wait) {
			first_wait = FALSE;
			printf("Waiting for DSMOS...\n");
		}
		return KERN_ABORTED;
	}
	return (*dsmos_hook)(from, to, src_offset, ops);
}


text_crypter_create_hook_t text_crypter_create;
void
text_crypter_create_hook_set(text_crypter_create_hook_t hook)
{
	text_crypter_create = hook;
}
