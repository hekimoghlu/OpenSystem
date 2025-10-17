/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#include "tkInt.h"


/*
 *----------------------------------------------------------------------
 *
 * TkpGetSystemDefault --
 *
 *	Given a dbName and className for a configuration option,
 *	return a string representation of the option.
 *
 * Results:
 *	Returns a Tk_Uid that is the string identifier that identifies
 *	this option. Returns NULL if there are no system defaults
 *	that match this pair.
 *
 * Side effects:
 *	None, once the package is initialized.
 *
 *----------------------------------------------------------------------
 */

Tcl_Obj *
TkpGetSystemDefault(
    Tk_Window tkwin,			/* A window to use. */
    CONST char *dbName,			/* The option database name. */
    CONST char *className)		/* The name of the option class. */
{
    return NULL;
}
