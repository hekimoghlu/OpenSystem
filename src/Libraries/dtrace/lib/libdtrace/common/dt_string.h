/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef	_DT_STRING_H
#define	_DT_STRING_H

#include <sys/types.h>
#include <strings.h>

#ifdef	__cplusplus
extern "C" {
#endif

extern char *strndup(const char *, size_t);
extern size_t stresc2chr(char *);
extern char *strchr2esc(const char *, size_t);
extern const char *strbasename(const char *);
extern const char *strbadidnum(const char *);
extern int strisglob(const char *);
extern char *strhyphenate(char *);

#ifdef	__cplusplus
}
#endif

#endif	/* _DT_STRING_H */
