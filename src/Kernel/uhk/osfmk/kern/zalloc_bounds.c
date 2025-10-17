/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
#include <kern/turnstile.h>
#include <kern/kalloc.h>

ZONE_DECLARE_ID(ZONE_ID_TURNSTILE, struct turnstile);

#if DEBUG || DEVELOPMENT
/*
 * Make sure the various allocators work with bound checking.
 */
extern void zalloc_bound_checks(void);
extern void kalloc_type_bound_checks(void);
extern void kalloc_data_bound_checks(void);

void
zalloc_bound_checks(void)
{
	struct turnstile *__single ts = zalloc_id(ZONE_ID_TURNSTILE, Z_WAITOK);

	zfree_id(ZONE_ID_TURNSTILE, ts);
}

void
kalloc_data_bound_checks(void)
{
	int *__single i;
	int *__bidi_indexable a;
	void *__bidi_indexable d;

	d = kalloc_data(10, Z_WAITOK);
	kfree_data(d, 10);

	i = kalloc_type(int, Z_WAITOK);
	kfree_type(int, i);

	a = kalloc_type(int, 10, Z_WAITOK);
	a = krealloc_type(int, 10, 20, a, Z_WAITOK | Z_REALLOCF);
	kfree_type(int, 20, a);

	a = kalloc_type(int, int, 10, Z_WAITOK);
	a = krealloc_type(int, int, 10, 20, a, Z_WAITOK | Z_REALLOCF);
	kfree_type(int, int, 20, a);
}

void
kalloc_type_bound_checks(void)
{
	struct turnstile *__single ts;
	struct turnstile *__bidi_indexable ts_a;

	ts = kalloc_type(struct turnstile, Z_WAITOK);

	kfree_type(struct turnstile, ts);

	ts_a = kalloc_type(struct turnstile, 10, Z_WAITOK);

	ts_a = krealloc_type(struct turnstile, 10, 20,
	    ts_a, Z_WAITOK | Z_REALLOCF);

	kfree_type(struct turnstile, 20, ts_a);

	ts_a = kalloc_type(struct turnstile, struct turnstile, 10, Z_WAITOK);

	ts_a = krealloc_type(struct turnstile, struct turnstile, 10, 20,
	    ts_a, Z_WAITOK | Z_REALLOCF);

	kfree_type(struct turnstile, struct turnstile, 20, ts_a);
}
#endif /* DEBUG || DEVELOPMENT */
