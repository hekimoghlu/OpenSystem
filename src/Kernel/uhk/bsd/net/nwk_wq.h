/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
#ifndef NWK_WQ_H
#define NWK_WQ_H
#include <sys/queue.h>
#include <kern/kern_types.h>
#include <os/base.h>

#ifdef BSD_KERNEL_PRIVATE
struct nwk_wq_entry {
	void(*XNU_PTRAUTH_SIGNED_FUNCTION_PTR("nkw_wq_entry.func") func)(struct nwk_wq_entry *);
	TAILQ_ENTRY(nwk_wq_entry) nwk_wq_link;
};

void nwk_wq_init(void);
void nwk_wq_enqueue(struct nwk_wq_entry *nwk_item);
#endif /* BSD_KERNEL_PRIVATE */
#endif /* NWK_WQ_H */
