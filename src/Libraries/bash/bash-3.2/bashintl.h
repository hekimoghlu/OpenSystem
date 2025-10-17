/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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
#if !defined (_BASHINTL_H_)
#define _BASHINTL_H_

#if defined (BUILDTOOL)
#  undef ENABLE_NLS
#  define ENABLE_NLS 0
#endif

/* Include this *after* config.h */
#include "gettext.h"

#if defined (HAVE_LOCALE_H)
#  include <locale.h>
#endif

#define _(msgid)	gettext(msgid)
#define N_(msgid)	msgid
#define D_(d, msgid)	dgettext(d, msgid)

#if defined (HAVE_SETLOCALE) && !defined (LC_ALL)
#  undef HAVE_SETLOCALE
#endif

#if !defined (HAVE_SETLOCALE)
#  define setlocale(cat, loc)
#endif

#endif /* !_BASHINTL_H_ */
