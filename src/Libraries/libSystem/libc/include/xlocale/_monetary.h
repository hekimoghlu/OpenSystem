/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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
#ifndef _XLOCALE__MONETARY_H_
#define _XLOCALE__MONETARY_H_

#include <sys/cdefs.h>
#include <_types.h>
#include <sys/_types/_size_t.h>
#include <sys/_types/_ssize_t.h>
#include <_xlocale.h>

__BEGIN_DECLS
ssize_t	strfmon_l(char *, size_t, locale_t, const char *, ...)
		__strfmonlike(4, 5);
__END_DECLS

#endif /* _XLOCALE__MONETARY_H_ */
