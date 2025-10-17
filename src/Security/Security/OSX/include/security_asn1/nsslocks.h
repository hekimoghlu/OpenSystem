/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
 * nsslocks.h - threadsafe functions to initialize lock pointers.
 *
 * NOTE - These are not public interfaces
 *
 * $Id: nsslocks.h,v 1.1 2003/01/30 22:42:07 dmitch Exp $
 */

#ifndef _NSSLOCKS_H_
#define _NSSLOCKS_H_

#include "seccomon.h"
#include "nssilock.h"
#include "prmon.h"

SEC_BEGIN_PROTOS

/* Given the address of a (global) pointer to a PZLock, 
 * atomicly create the lock and initialize the (global) pointer, 
 * if it is not already created/initialized.
 */

extern SECStatus nss_InitLock(   PZLock    **ppLock, nssILockType ltype );

/* Given the address of a (global) pointer to a PZMonitor, 
 * atomicly create the monitor and initialize the (global) pointer, 
 * if it is not already created/initialized.
 */

extern SECStatus nss_InitMonitor(PZMonitor **ppMonitor, nssILockType ltype );

SEC_END_PROTOS

#endif
