/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 16, 2023.
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
 * This is called from sys/select.h and sys/time.h for the common prototype
 * of select().  Setting _DARWIN_C_SOURCE or _DARWIN_UNLIMITED_SELECT uses
 * the version of select() that does not place a limit on the first argument
 * (nfds).  In the UNIX conformance case, values of nfds greater than
 * FD_SETSIZE will return an error of EINVAL.
 */
#ifndef _SYS__SELECT_H_
#define _SYS__SELECT_H_

#include <sys/cdefs.h> /* __DARWIN_EXTSN_C, __DARWIN_1050, __DARWIN_ALIAS_C */
#include <sys/_types/_fd_def.h> /* fd_set */
#include <sys/_types/_timeval.h> /* struct timeval */

int      select(int, fd_set * __restrict, fd_set * __restrict,
    fd_set * __restrict, struct timeval * __restrict)

#if defined(_DARWIN_C_SOURCE) || defined(_DARWIN_UNLIMITED_SELECT)
__DARWIN_EXTSN_C(select)
#else /* !_DARWIN_C_SOURCE && !_DARWIN_UNLIMITED_SELECT */
#  if defined(__LP64__) && !__DARWIN_NON_CANCELABLE
__DARWIN_1050(select)
#  else /* !__LP64__ || __DARWIN_NON_CANCELABLE */
__DARWIN_ALIAS_C(select)
#  endif /* __LP64__ && !__DARWIN_NON_CANCELABLE */
#endif /* _DARWIN_C_SOURCE || _DARWIN_UNLIMITED_SELECT */
;

#endif /* !_SYS__SELECT_H_ */
