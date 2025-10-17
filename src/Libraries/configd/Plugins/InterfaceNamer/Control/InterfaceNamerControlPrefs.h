/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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
#ifndef _INTERFACENAMERCONTROLPREFS_H
#define _INTERFACENAMERCONTROLPREFS_H

/*
 * InterfaceNamerControlPrefs.h
 * - definitions for accessing InterfaceNamer control preferences
 */

/*
 * Modification History
 *
 * January 12, 2017	Allan Nathanson (ajn@apple.com)
 * - created
 */

#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include "SCControlPrefs.h"

__BEGIN_DECLS

typedef void (*InterfaceNamerControlPrefsCallBack)(_SCControlPrefsRef control);

_SCControlPrefsRef
InterfaceNamerControlPrefsInit(CFRunLoopRef runloop,
			       InterfaceNamerControlPrefsCallBack callback);

Boolean
InterfaceNamerControlPrefsAllowNewInterfaces(_SCControlPrefsRef control);

Boolean
InterfaceNamerControlPrefsSetAllowNewInterfaces(_SCControlPrefsRef control,
						Boolean allow);

Boolean
InterfaceNamerControlPrefsConfigureNewInterfaces(_SCControlPrefsRef control);

Boolean
InterfaceNamerControlPrefsSetConfigureNewInterfaces(_SCControlPrefsRef control,
						    Boolean configure);

__END_DECLS

#endif	/* _INTERFACENAMERCONTROLPREFS_H */
