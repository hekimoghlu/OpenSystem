/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
 * Copyright (c) 2011 Uwe Stuehler <uwe@openbsd.org>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <sys/param.h>

#include <machine/bus.h>

#include <armv7/armv7/armv7var.h>

#define OMAPID_ADDR	0x4a002000
#define OMAPID_SIZE	0x1000

#define PRM_ADDR	0x4a306000
#define PRM_SIZE	0x2000
#define CM1_ADDR	0x4a004000
#define CM1_SIZE	0x1000
#define CM2_ADDR	0x4a008000
#define CM2_SIZE	0x2000

struct armv7_dev omap4_devs[] = {

	/*
	 * Power, Reset and Clock Manager
	 */

	{ .name = "prcm",
	  .unit = 0,
	  .mem = {
	    { PRM_ADDR, PRM_SIZE },
	    { CM1_ADDR, CM1_SIZE },
	    { CM2_ADDR, CM2_SIZE },
	  },
	},

	/*
	 * OMAP identification registers/fuses
	 */

	{ .name = "omapid",
	  .unit = 0,
	  .mem = { { OMAPID_ADDR, OMAPID_SIZE } },
	},

	/* Terminator */
	{ .name = NULL,
	  .unit = 0,
	}
};

void
omap4_init(void)
{
	armv7_set_devs(omap4_devs);
}

