/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 27, 2023.
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
#ifndef _SCPREFS_OBSERVER_H
#define _SCPREFS_OBSERVER_H

#include <os/availability.h>
#include <TargetConditionals.h>
#include <sys/cdefs.h>
#include <dispatch/dispatch.h>

typedef enum {
#if	!TARGET_OS_IPHONE
		scprefs_observer_type_mcx	= 1,
#else	// !TARGET_OS_IPHONE
		scprefs_observer_type_global	= 2,
#endif	// !TARGET_OS_IPHONE
} _scprefs_observer_type;

typedef struct _scprefs_observer_t * scprefs_observer_t;

__BEGIN_DECLS

/*!
 @function prefs_observer_watch
 @discussion Sends a notification to interested configuration agents
 when a particular preference file has changed.
 @param type the type of preference (MCX on OSX, Global/Profiles on iOS) to watch.
 @param plist_name the name of the plist file to watch.
 @param queue the queue to be called back on.
 @param block the block to be called back on.
 @result Returns the created preferences observer
 */
scprefs_observer_t
_scprefs_observer_watch(_scprefs_observer_type type, const char *plist_name,
			dispatch_queue_t queue, dispatch_block_t block);

/*!
 @function prefs_observer_watch
 @discussion Cancels/deregisters the given preferences watcher.
 @param observer the watcher to be cancelled.
 */
void
_scprefs_observer_cancel(scprefs_observer_t observer);

__END_DECLS

#endif	/* _SCPREFS_OBSERVER_H */
