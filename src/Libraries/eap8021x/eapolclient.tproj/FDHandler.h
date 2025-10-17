/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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

#ifndef _S_FDHANDLER_H
#define _S_FDHANDLER_H

/*
 * Copyright (c) 2000-2008 Apple Inc. All rights reserved.
 *
 * @APPLE_LICENSE_HEADER_START@
 * 
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 * 
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 * 
 * @APPLE_LICENSE_HEADER_END@
 */

/*
 * FDHandler.h
 * - notification/callback services for a file descriptor
 */

/* 
 * Modification History
 *
 * October 26, 2001	Dieter Siegmund (dieter@apple.com)
 * - created (based on bootp/IPConfiguration.tproj/FDSet.c)
 */

/*
 * Type: FDHandler_func
 * Purpose:
 *   Function to call when the file descriptor is ready.
 */

typedef void (FDHandler_func)(void * arg1, void * arg2);

typedef struct FDHandler_s FDHandler;

FDHandler * 
FDHandler_create(int fd);

void
FDHandler_free(FDHandler * * handler_p);

void
FDHandler_enable(FDHandler * handler, FDHandler_func * func, 
		 void * arg1, void * arg2);
void
FDHandler_disable(FDHandler * handler);

int
FDHandler_fd(FDHandler * handler);

#endif /* _S_FDHANDLER_H */
