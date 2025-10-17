/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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
__FBSDID("$FreeBSD: src/lib/libc/stdio/getwchar.c,v 1.3 2004/05/25 10:42:52 tjr Exp $");

#include "xlocale_private.h"

#include "namespace.h"
#include <stdio.h>
#include <wchar.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"

#undef getwchar

/*
 * Synonym for fgetwc(stdin).
 */
wint_t
getwchar(void)
{

	return (fgetwc_l(stdin, __current_locale()));
}

wint_t
getwchar_l(locale_t loc)
{

	/* no need to call NORMALIZE_LOCALE(loc) because fgetwc_l will */
	return (fgetwc_l(stdin, loc));
}
