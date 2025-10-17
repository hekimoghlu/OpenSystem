/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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
 * Copyright (c) 1988 NeXT, Inc.
 *
 * HISTORY
 *  04-May-90 Created
 */

#ifndef __MONITOR_HEADER__
#define __MONITOR_HEADER__

#include <sys/cdefs.h>
#include <Availability.h>

__BEGIN_DECLS

__OSX_AVAILABLE_BUT_DEPRECATED_MSG(__MAC_10_0,__MAC_10_11,__IPHONE_2_0,__IPHONE_9_0, "Monitor is no longer supported.")
__WATCHOS_PROHIBITED
extern void monstartup (char *lowpc, char *highpc);

__OSX_AVAILABLE_BUT_DEPRECATED_MSG(__MAC_10_0,__MAC_10_11,__IPHONE_2_0,__IPHONE_9_0, "Monitor is no longer supported.")
__WATCHOS_PROHIBITED
extern void monitor (char *lowpc, char *highpc, char *buf, int bufsiz, int cntsiz);

__OSX_AVAILABLE_BUT_DEPRECATED_MSG(__MAC_10_0,__MAC_10_11,__IPHONE_2_0,__IPHONE_9_0, "Monitor is no longer supported.")
__WATCHOS_PROHIBITED
extern void moncontrol (int mode);

__OSX_AVAILABLE_BUT_DEPRECATED_MSG(__MAC_10_0,__MAC_10_11,__IPHONE_2_0,__IPHONE_9_0, "Monitor is no longer supported.")
__WATCHOS_PROHIBITED
extern void monoutput (const char *filename);

__OSX_AVAILABLE_BUT_DEPRECATED_MSG(__MAC_10_0,__MAC_10_11,__IPHONE_2_0,__IPHONE_9_0, "Monitor is no longer supported.")
__WATCHOS_PROHIBITED
extern void moninit (void);

__OSX_AVAILABLE_BUT_DEPRECATED_MSG(__MAC_10_0,__MAC_10_11,__IPHONE_2_0,__IPHONE_9_0, "Monitor is no longer supported.")
__WATCHOS_PROHIBITED
extern void monreset (void);

__END_DECLS

#endif	/* __MONITOR_HEADER__ */
