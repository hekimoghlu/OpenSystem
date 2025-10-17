/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/locale/setinvalidrune.c,v 1.3 2002/09/24 09:25:37 tjr Exp $");

#include "xlocale_private.h"

#include <rune.h>
#include "mblocal.h"
#include "runedepreciated.h"

void
setinvalidrune(rune_t ir)
{
	struct xlocale_ctype *rl;
	static int warn_depreciated = 1;
	locale_t loc = __current_locale();

	if (warn_depreciated) {
		warn_depreciated = 0;
		fprintf(stderr, __rune_depreciated_msg, "setinvalidrune");
	}

	rl = (void *)loc->components[XLC_CTYPE];
	if (rl->_CurrentRuneLocale->__invalid_rune != ir) {
		struct xlocale_ctype *new = (struct xlocale_ctype *)malloc(rl->__datasize);
		if (!new)
			return;
		*new = *rl;
		new->header.header.retain_count = 1;
		new->_CurrentRuneLocale->__invalid_rune = ir;
		xlocale_release(rl);
		loc->components[XLC_CTYPE] = (void *)new;
	}
}
