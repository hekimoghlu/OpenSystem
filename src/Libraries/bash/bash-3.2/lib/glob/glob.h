/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#ifndef	_GLOB_H_
#define	_GLOB_H_

#include "stdc.h"

#define GX_MARKDIRS	0x01	/* mark directory names with trailing `/' */
#define GX_NOCASE	0x02	/* ignore case */
#define GX_MATCHDOT	0x04	/* match `.' literally */

extern int glob_pattern_p __P((const char *));
extern char **glob_vector __P((char *, char *, int));
extern char **glob_filename __P((char *, int));

extern char *glob_error_return;
extern int noglob_dot_filenames;
extern int glob_ignore_case;

#endif /* _GLOB_H_ */
