/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
 * Bootparams lookup routines
 *
 * Copyright 1997
 * Apple Computer Inc.
 */

#ifndef _BOOTPARAMS_H_
#define _BOOTPARAMS_H_

#include <sys/param.h>
#include <sys/cdefs.h>

/*
 * Structures returned by bootparams calls.
 */
struct bootparamsent {
	char *bp_name;			/* name of host */
	char **bp_bootparams;	/* bootparams list */
};

__BEGIN_DECLS
void bootparams_endent __P((void));
struct bootparamsent *bootparams_getbyname __P((const char *));
struct bootparamsent *bootparams_getent __P((void));
void bootparams_setent __P((void));
__END_DECLS

#endif /* !_BOOTPARAMS_H_ */
