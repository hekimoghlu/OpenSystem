/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#ifndef _SCPREFSCONTROL_H
#define _SCPREFSCONTROL_H

/*
 * SCControlPrefs.h
 * - APIs for accessing control preferences and being notified
 *   when they change
 */

/*
 * Modification History
 *
 * Jun 10, 2021			Allan Nathanson (ajn@apple.com)
 * - created
 */

#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>

__BEGIN_DECLS

typedef struct CF_BRIDGED_TYPE(id) __SCControlPrefs * _SCControlPrefsRef;

typedef void (*_SCControlPrefsCallBack)			(_SCControlPrefsRef		control);

_SCControlPrefsRef	_SCControlPrefsCreate		(const char			*prefsPlist,
							 CFRunLoopRef			runloop,
							 _SCControlPrefsCallBack	callback);

_SCControlPrefsRef	_SCControlPrefsCreateWithQueue	(const char			*prefsPlist,
							 dispatch_queue_t		queue,
							 _SCControlPrefsCallBack	callback);

Boolean			_SCControlPrefsGetBoolean	(_SCControlPrefsRef		control,
							 CFStringRef			key);

Boolean			_SCControlPrefsSetBoolean	(_SCControlPrefsRef		control,
							 CFStringRef			key,
							 Boolean			enabled);

__END_DECLS

#endif	/* _SCPREFSCONTROL_H */
