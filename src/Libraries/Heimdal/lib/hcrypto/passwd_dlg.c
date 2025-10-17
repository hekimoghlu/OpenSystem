/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
/* passwd_dlg.c - Dialog boxes for Windows95/NT
 * Author:	JÃ¶rgen Karlsson - d93-jka@nada.kth.se
 * Date:	June 1996
 */

#include <config.h>

#ifdef WIN32	/* Visual C++ 4.0 (Windows95/NT) */
#include <Windows.h>
#include "passwd_dlg.h"
#include "Resource.h"
#define passwdBufSZ 64

char passwd[passwdBufSZ];

BOOL CALLBACK
pwd_dialog_proc(HWND hwndDlg, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch(uMsg)
    {
    case WM_COMMAND:
	switch(wParam)
	{
	case IDOK:
	    if(!GetDlgItemText(hwndDlg,IDC_PASSWD_EDIT, passwd, passwdBufSZ))
		EndDialog(hwndDlg, IDCANCEL);
	case IDCANCEL:
	    EndDialog(hwndDlg, wParam);
	    return TRUE;
	}
    }
    return FALSE;
}


/* return 0 if ok, 1 otherwise */
int
pwd_dialog(char *buf, int size)
{
    int i;
    HWND wnd = GetActiveWindow();
    HANDLE hInst = GetModuleHandle("des");
    switch(DialogBox(hInst,MAKEINTRESOURCE(IDD_PASSWD_DIALOG),wnd,pwd_dialog_proc))
    {
    case IDOK:
	strlcpy(buf, passwd, size);
	memset (passwd, 0, sizeof(passwd));
	return 0;
    case IDCANCEL:
    default:
	memset (passwd, 0, sizeof(passwd));
	return 1;
    }
}

#endif /* WIN32 */
