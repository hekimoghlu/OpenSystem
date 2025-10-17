/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
 * Modification History
 *
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * November 9, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#ifndef _NOTIFICATIONS_H
#define _NOTIFICATIONS_H

#include <sys/cdefs.h>

__BEGIN_DECLS

void	storeCallback		(SCDynamicStoreRef	store,
				 CFArrayRef		changedKeys,
				 void			*info);

void	do_notify_list		(int argc, char * const argv[]);
void	do_notify_add		(int argc, char * const argv[]);
void	do_notify_remove	(int argc, char * const argv[]);
void	do_notify_changes	(int argc, char * const argv[]);
void	do_notify_watch		(int argc, char * const argv[]);
void	do_notify_wait		(int argc, char * const argv[]);
void	do_notify_file		(int argc, char * const argv[]);
void	do_notify_cancel	(int argc, char * const argv[]);

__END_DECLS

#endif	/* !_NOTIFICATIONS_H */
