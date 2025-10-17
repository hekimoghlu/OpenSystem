/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
 * Mach Operating System
 * Copyright (c) 1987 Carnegie-Mellon University
 * All rights reserved.  The CMU software License Agreement specifies
 * the terms and conditions for use and redistribution.
 */
/*
 *	File:	vm_pager_xnu.h
 *	Author:	Avadis Tevanian, Jr., Michael Wayne Young
 *
 *	Copyright (C) 1986, Avadis Tevanian, Jr., Michael Wayne Young
 *	Copyright (C) 1985, Avadis Tevanian, Jr., Michael Wayne Young
 *
 *	Pager routine interface definition
 */

#ifndef _VM_PAGER_
#define _VM_PAGER_

#include <mach/boolean.h>

struct  pager_struct {
	boolean_t       is_device;
};
typedef struct pager_struct     *vm_pager_t;
#define vm_pager_null           ((vm_pager_t) 0)

#define PAGER_SUCCESS           0  /* page read or written */
#define PAGER_ABSENT            1  /* pager does not have page */
#define PAGER_ERROR             2  /* pager unable to read or write page */

#endif  /* _VM_PAGER_ */
