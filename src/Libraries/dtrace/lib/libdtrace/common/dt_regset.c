/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
/*
 * Copyright 2003 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*
 * Copyright (c) 2012 by Delphix. All rights reserved.
 * Copyright (c) 2016 Pedro Giffuni.  All rights reserved.
 */

#include <sys/types.h>
#include <sys/bitmap.h>
#include <assert.h>
#include <strings.h>
#include <stdlib.h>

#include <dt_regset.h>
#include <dt_impl.h>

dt_regset_t *
dt_regset_create(ulong_t nregs)
{
	ulong_t n = BT_BITOUL(nregs);
	dt_regset_t *drp = malloc(sizeof (dt_regset_t));

	if (drp == NULL)
		return (NULL);

	drp->dr_bitmap = calloc(n, sizeof (ulong_t));

	if (drp->dr_bitmap == NULL) {
		dt_regset_destroy(drp);
		return (NULL);
	}

	drp->dr_size = nregs;

	return (drp);
}

void
dt_regset_destroy(dt_regset_t *drp)
{
	free(drp->dr_bitmap);
	free(drp);
}

void
dt_regset_reset(dt_regset_t *drp)
{
	bzero(drp->dr_bitmap, sizeof (ulong_t) * BT_BITOUL(drp->dr_size));
}

void
dt_regset_assert_free(dt_regset_t *drp)
{
	int reg;
	boolean_t fail = B_FALSE;
	for (reg = 0; reg < drp->dr_size; reg++) {
		if (BT_TEST(drp->dr_bitmap, reg) != 0)  {
			dt_dprintf("%%r%d was left allocated", reg);
			fail = B_TRUE;
		}
	}

	/*
	 * We set this during dtest runs to check for register leaks.
	 */
	if (fail && getenv("DTRACE_DEBUG_REGSET") != NULL)
		abort();
}

int
dt_regset_alloc(dt_regset_t *drp)
{
	ulong_t nbits = drp->dr_size - 1;
	ulong_t maxw = nbits >> BT_ULSHIFT;
	ulong_t wx;

	for (wx = 0; wx <= maxw; wx++) {
		if (drp->dr_bitmap[wx] != ~0UL)
			break;
	}

	if (wx <= maxw) {
		ulong_t maxb = (wx == maxw) ? nbits & BT_ULMASK : BT_NBIPUL - 1;
		ulong_t word = drp->dr_bitmap[wx];
		ulong_t bit, bx;
		int reg;

		for (bit = 1, bx = 0; bx <= maxb; bx++, bit <<= 1) {
			if ((word & bit) == 0) {
				reg = (int)((wx << BT_ULSHIFT) | bx);
				BT_SET(drp->dr_bitmap, reg);
				return (reg);
			}
		}
	}

	xyerror(D_NOREG, "Insufficient registers to generate code");
	/*NOTREACHED*/
	return (-1);
}

void
dt_regset_free(dt_regset_t *drp, int reg)
{
	assert(reg >= 0 && reg < drp->dr_size);
	assert(BT_TEST(drp->dr_bitmap, reg) != 0);
	BT_CLEAR(drp->dr_bitmap, reg);
}
