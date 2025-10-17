/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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
#include "../generic/tclThread.h"
#include <windows.h>
#include <process.h>

#if 0
/* only Windows 2000 (XP, too??) has this function */
HANDLE (WINAPI *winOpenThreadProc)(DWORD, BOOL, DWORD);

void
ThreadpInit (void)
{
    HMODULE hKernel = GetModuleHandle("kernel32.dll");
    winOpenThreadProc = (HANDLE (WINAPI *)(DWORD, BOOL, DWORD))
	    GetProcAddress(hKernel, "OpenThread");
}

int
ThreadpKill (Tcl_Interp *interp, long id)
{
    HANDLE hThread;
    int result = TCL_OK;

    if (winOpenThreadProc) {
	hThread = winOpenThreadProc(THREAD_TERMINATE, FALSE, id);
	/* 
	 * not to be misunderstood as "devilishly clever",
	 * but evil in it's pure form.
	 */
	TerminateThread(hThread, 666);
    } else {
	Tcl_AppendStringsToObj(Tcl_GetObjResult(interp),
		"Can't (yet) kill threads on this OS, sorry.", NULL);
	result = TCL_ERROR;
    }
    return result;
}
#endif
