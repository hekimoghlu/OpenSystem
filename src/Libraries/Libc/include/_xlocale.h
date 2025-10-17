/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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
#ifndef _XLOCALE_H_
#define _XLOCALE_H_

#include <sys/cdefs.h>
#include <_bounds.h>

#ifndef _USE_EXTENDED_LOCALES_
#define _USE_EXTENDED_LOCALES_
#endif /* _USE_EXTENDED_LOCALES_ */

#include <_locale.h>
#include <__xlocale.h>

_LIBC_SINGLE_BY_DEFAULT()

__BEGIN_DECLS
extern const locale_t _c_locale;

struct lconv *	localeconv_l(locale_t);
__const char *	querylocale(int, locale_t);
__END_DECLS

#endif /* _XLOCALE_H_ */
