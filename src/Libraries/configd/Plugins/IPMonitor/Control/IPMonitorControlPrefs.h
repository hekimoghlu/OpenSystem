/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
#ifndef _IPMONITORCONTROLPREFS_H
#define _IPMONITORCONTROLPREFS_H

/*
 * IPMonitorControlPrefs.h
 * - definitions for accessing IPMonitor control preferences and being notified
 *   when they change
 */

/*
 * Modification History
 *
 * January 14, 2013	Dieter Siegmund (dieter@apple)
 * - created
 */

#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include "SCControlPrefs.h"

__BEGIN_DECLS

typedef void (*IPMonitorControlPrefsCallBack)(_SCControlPrefsRef control);

_SCControlPrefsRef
IPMonitorControlPrefsInit	(dispatch_queue_t queue,
				 IPMonitorControlPrefsCallBack	callback);

Boolean
IPMonitorControlPrefsIsVerbose	(void);

Boolean
IPMonitorControlPrefsSetVerbose	(Boolean			verbose);

Boolean
IPMonitorControlPrefsGetDisableServiceCoupling(void);

Boolean
IPMonitorControlPrefsSetDisableServiceCoupling(Boolean disable_coupling);

__END_DECLS

#endif	/* _IPMONITORCONTROLPREFS_H */
