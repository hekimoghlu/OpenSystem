/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#ifdef _MSC_VER
    /* Only do this when MSVC++ is compiling us. */
#   define DllEntryPoint DllMain
#   if defined(USE_TCL_STUBS) && (!defined(_MT) || !defined(_DLL) || defined(_DEBUG))
	/*
	 * This fixes a bug with how the Stubs library was compiled.
	 * The requirement for msvcrt.lib from tclstubXX.lib should
	 * be removed.
	 */
#	pragma comment(linker, "-nodefaultlib:msvcrt.lib")
#   endif
#endif

/*
 *----------------------------------------------------------------------
 *
 * DllEntryPoint --
 *
 *	This wrapper function is used by Windows to invoke the
 *	initialization code for the DLL.  If we are compiling
 *	with Visual C++, this routine will be renamed to DllMain.
 *
 * Results:
 *	Returns TRUE;
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

#ifndef STATIC_BUILD

BOOL APIENTRY
DllEntryPoint(hInst, reason, reserved)
    HINSTANCE hInst;		/* Library instance handle. */
    DWORD reason;		/* Reason this function is being called. */
    LPVOID reserved;		/* Not used. */
{
    return TRUE;
}

#endif