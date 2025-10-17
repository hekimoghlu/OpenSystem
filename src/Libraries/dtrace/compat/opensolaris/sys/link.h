/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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
 *	Copyright (c) 1988 AT&T
 *	  All Rights Reserved
 *
 * Copyright 2008 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef _SYS_LINK_H
#define	_SYS_LINK_H

#ifndef	_ASM
#include <sys/types.h>

#include <sys/elftypes.h>

#endif

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * Public structure defined and maintained within the runtime linker
 */
#ifndef	_ASM
/*
 * Debugging events enabled inside of the runtime linker.  To
 * access these events see the librtld_db interface.
 */
typedef enum {
	RD_NONE = 0,		/* no event */
	RD_PREINIT,		/* the Initial rendezvous before .init */
	RD_POSTINIT,		/* the Second rendezvous after .init */
	RD_DLACTIVITY,		/* a dlopen or dlclose has happened */
	RD_DYLD_LOST,		/* communication with target dyld was lost */
	RD_DYLD_EXIT		/* target exited. */
} rd_event_e;

#endif	/* _ASM */

#ifdef	__cplusplus
}
#endif

#endif	/* _SYS_LINK_H */
