/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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
 * InterfaceNamerControlPrefs.c
 * - definitions for accessing InterfaceNamer control preferences
 */

/*
 * Modification History
 *
 * January 12, 2017	Allan Nathanson (ajn@apple.com)
 * - created
 */

#include "InterfaceNamerControlPrefs.h"

/*
 * kInterfaceNamerControlPrefsID
 * - identifies the InterfaceNamer preferences file that contains 'AllowNewInterfaces'
 */
#define kInterfaceNamerControlPrefsIDStr	"com.apple.InterfaceNamer.control.plist"

/*
 * kAllowNewInterfaces
 * - indicates whether InterfaceNamer is allowed to create new interfaces
 *   while the screen is locked or not
 */
#define kAllowNewInterfaces			CFSTR("AllowNewInterfaces")

/*
 * kConfigureNewInterfaces
 * - indicates whether InterfaceNamer should configure new interfaces as they
 *   appear
 */
#define kConfigureNewInterfaces			CFSTR("ConfigureNewInterfaces")

__private_extern__
_SCControlPrefsRef
InterfaceNamerControlPrefsInit(CFRunLoopRef				runloop,
			       InterfaceNamerControlPrefsCallBack	callback)
{
	_SCControlPrefsRef	control;

	control = _SCControlPrefsCreate(kInterfaceNamerControlPrefsIDStr,
					runloop, callback);
	return control;
}

/**
 ** Get
 **/
static Boolean
get_prefs_bool(_SCControlPrefsRef control, CFStringRef key)
{
	Boolean	val	= FALSE;

	if (control != NULL) {
		val = _SCControlPrefsGetBoolean(control, key);
	}
	return val;
}

__private_extern__ Boolean
InterfaceNamerControlPrefsAllowNewInterfaces(_SCControlPrefsRef control)
{
	return get_prefs_bool(control, kAllowNewInterfaces);
}

__private_extern__ Boolean
InterfaceNamerControlPrefsConfigureNewInterfaces(_SCControlPrefsRef control)
{
	return get_prefs_bool(control, kConfigureNewInterfaces);
}

/**
 ** Set
 **/
static Boolean
set_prefs_bool(_SCControlPrefsRef control, CFStringRef key, Boolean val)
{
	Boolean	ok	= FALSE;

	if (control != NULL) {
		ok = _SCControlPrefsSetBoolean(control, key, val);
	}
	return ok;
}

__private_extern__ Boolean
InterfaceNamerControlPrefsSetAllowNewInterfaces(_SCControlPrefsRef control,
						Boolean allow)
{
	return set_prefs_bool(control, kAllowNewInterfaces, allow);
}

Boolean
InterfaceNamerControlPrefsSetConfigureNewInterfaces(_SCControlPrefsRef control,
						    Boolean configure)
{
	return set_prefs_bool(control, kConfigureNewInterfaces, configure);
}
