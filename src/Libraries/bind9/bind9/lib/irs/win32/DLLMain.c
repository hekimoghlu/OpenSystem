/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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
/* $Id$ */

#include <windows.h>
#include <signal.h>

/*
 * Called when we enter the DLL
 */
__declspec(dllexport) BOOL WINAPI DllMain(HINSTANCE hinstDLL,
					  DWORD fdwReason, LPVOID lpvReserved)
{
	switch (fdwReason) {
	/*
	 * The DLL is loading due to process
	 * initialization or a call to LoadLibrary.
	 */
	case DLL_PROCESS_ATTACH:
		break;

	/* The attached process creates a new thread.  */
	case DLL_THREAD_ATTACH:
		break;

	/* The thread of the attached process terminates. */
	case DLL_THREAD_DETACH:
		break;

	/*
	 * The DLL is unloading from a process due to
	 * process termination or a call to FreeLibrary.
	 */
	case DLL_PROCESS_DETACH:
		break;

	default:
		break;
	}
	return (TRUE);
}

