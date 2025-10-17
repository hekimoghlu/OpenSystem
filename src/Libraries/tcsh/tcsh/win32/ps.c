/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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
 * ps.c : ps,shutdown builtins.
 * -amol
 *
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winperf.h>
#include <tlhelp32.h>
#include <sh.h>
#include <errno.h>

static HWND ghwndtokillbywm_close;
BOOL CALLBACK enum_wincb2(HWND hwnd,LPARAM pidtokill) {
	DWORD pid = 0;

	if (!GetWindowThreadProcessId(hwnd,&pid))
		return TRUE;
	if (pid == (DWORD)pidtokill){
		ghwndtokillbywm_close = hwnd;
		PostMessage( hwnd, WM_CLOSE, 0, 0 );
		return TRUE;
	}

	return TRUE;
}
int kill_by_wm_close(int pid)  {
	EnumWindows(enum_wincb2,(LPARAM)pid);
	if (!ghwndtokillbywm_close)
		return -1;
	ghwndtokillbywm_close = NULL;
	return 0;
}
