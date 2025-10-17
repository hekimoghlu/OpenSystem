/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
 * nt.char.c : Does NLS-like stuff
 * -amol
 *
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include "sh.h"


static HMODULE hlangdll;


extern DWORD gdwPlatform;


unsigned char oem_it(unsigned char ch) {
	unsigned char ch1[2],ch2[2];

	ch1[0] = ch;
	ch1[1] = 0;

	OemToChar((char*)ch1,(char*)ch2);

	return ch2[0];
}
void nls_dll_unload(void) {
	FreeLibrary(hlangdll);
	hlangdll=NULL;
}
char * nt_cgets(int set, int msgnum, char *def) {

	int rc;
	int msg;
	static char oembuf[256];/*FIXBUF*/
	WCHAR buffer[256];/*FIXBUF*/



	if (!hlangdll)
		return def;

	msg = set * 10000 + msgnum;

	if (gdwPlatform == VER_PLATFORM_WIN32_WINDOWS) {
		rc = LoadString(hlangdll,msg,oembuf,sizeof(oembuf));

		if(!rc)
			return def;
		return oembuf;
	}
	rc = LoadStringW(hlangdll,msg,buffer,ARRAYSIZE(buffer));

	if(!rc)
		return def;

	WideCharToMultiByte(CP_OEMCP,
			0,
			buffer,
			-1,
			oembuf,//winbuf,
			256,
			NULL,NULL);

	return oembuf;
}
#if defined(DSPMBYTE)
void nt_autoset_dspmbyte(void) {
	switch (GetConsoleCP()) {
		case 932: /* Japan */
			setcopy(CHECK_MBYTEVAR, STRsjis, VAR_READWRITE);
			update_dspmbyte_vars();
			break;
	}
}

// _mbmap must be copied to the child during fork()
unsigned short _mbmap[256] = { 0 };
#endif

#undef free
void nls_dll_init(void) {

	char *ptr;
	size_t size = 0;


	if (_dupenv_s(&ptr,&size,"TCSHLANG") == 0){

		if (hlangdll)
			FreeLibrary(hlangdll);
		hlangdll = LoadLibrary(ptr);

		free(ptr);
	}
}
