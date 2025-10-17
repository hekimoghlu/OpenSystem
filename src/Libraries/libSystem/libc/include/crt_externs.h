/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
 * Copyright (c) 1995 NeXT Computer, Inc. All Rights Reserved
 */

/*
** Prototypes for the functions to get environment information in
** the world of dynamic libraries. Lifted from .c file of same name.
** Fri Jun 23 12:56:47 PDT 1995
** AOF (afreier@next.com)
*/

#include <sys/cdefs.h>

__BEGIN_DECLS
extern char ***_NSGetArgv(void);
extern int *_NSGetArgc(void);
extern char ***_NSGetEnviron(void);
extern char **_NSGetProgname(void);
#ifdef __LP64__
extern struct mach_header_64 *
#else /* !__LP64__ */
extern struct mach_header *
#endif /* __LP64__ */
				_NSGetMachExecuteHeader(void);
__END_DECLS
