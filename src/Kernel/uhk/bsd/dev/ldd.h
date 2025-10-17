/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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
/*	@(#)ldd.h	2.0	03/20/90	(c) 1990 NeXT
 *
 * ldd.h - kernel prototypes used by loadable device drivers
 *
 * HISTORY
 * 22-May-91	Gregg Kellogg (gk) at NeXT
 *	Split out public interface.
 *
 * 16-Aug-90  Gregg Kellogg (gk) at NeXT
 *	Removed a lot of stuff that's defined in other header files.
 *	Eventually this file should either go away or contain only imports of
 *	other files.
 *
 * 20-Mar-90	Doug Mitchell at NeXT
 *	Created.
 *
 */

#ifndef _BSD_DEV_LDD_PRIV_
#define _BSD_DEV_LDD_PRIV_

#include <sys/cdefs.h>
#include <sys/disk.h>

typedef int (*PFI)();

#endif  /* _BSD_DEV_LDD_PRIV_ */
