/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
#include <windows.h>

#if defined(_MSC_VER) || defined(__GNUC__)
#define DllEntryPoint DllMain
#endif

/*
 *----------------------------------------------------------------------
 *
 * DllEntryPoint --
 *
 *	This routine is called by gcc, VC++ or Borland to invoke the
 *	initialization code for Tcl.
 *
 * Results:
 *	TRUE.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

BOOL APIENTRY
DllEntryPoint (
    HINSTANCE hInst,		/* Library instance handle. */
    DWORD reason,		/* Reason this function is being called. */
    LPVOID reserved)		/* Not used. */
{
    return TRUE; 
}
