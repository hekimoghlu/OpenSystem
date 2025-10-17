/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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
 * globals.c: The mem locations needed in the child are copied here.
 * -amol
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#define STRSAFE_LIB
#define STRSAFE_NO_CCH_FUNCTIONS
#include <strsafe.h>

extern unsigned long bookend1,bookend2;
extern char **environ;

#define IMAGE_SIZEOF_NT_OPTIONAL32_HEADER    224
#define IMAGE_SIZEOF_NT_OPTIONAL64_HEADER    240

#ifdef _WIN64
#define IMAGE_SIZEOF_NT_OPTIONAL_HEADER     IMAGE_SIZEOF_NT_OPTIONAL64_HEADER
#else
#define IMAGE_SIZEOF_NT_OPTIONAL_HEADER     IMAGE_SIZEOF_NT_OPTIONAL32_HEADER
#endif


#undef dprintf
void
dprintf(char *format, ...)
{				/* } */
	va_list vl;
	char putbuf[2048];
	DWORD err;

	err = GetLastError();
	{
		va_start(vl, format);
#pragma warning(disable:4995)
		wvsprintf(putbuf,format, vl);
#pragma warning(default:4995)
		va_end(vl);
		OutputDebugString(putbuf);
	}
	SetLastError(err);
}
/*
 * This function is called by fork(). The process must copy
 * whatever memory is needed in the child. hproc is a handle
 * to the child process
 *
 */
int fork_copy_user_mem(HANDLE hproc) {
	
	SIZE_T bytes,rc;
	SIZE_T size;
	void *low = &bookend1, *high= &bookend2;

	if(&bookend1 > &bookend2) {
		low = &bookend2;
		high = &bookend1;
	}

	size =(char*)high - (char*)low;


	rc =WriteProcessMemory(hproc,low,low, (DWORD)size, &bytes);

	if (!rc) {
		rc = GetLastError();
		return -1;
	}
	if (size != bytes) {
		//dprintf("size %d , wrote %d\n",size,bytes);
	}
	return 0;
}
/*
 * Inspired by Microsoft KB article ID: Q90493 
 *
 * returns 0 (false) if app is non-gui, 1 otherwise.
*/
#include <winnt.h>
#include <ntport.h>

__inline BOOL wait_for_io(HANDLE hi, OVERLAPPED *pO) {

        DWORD bytes = 0;
        if(GetLastError() != ERROR_IO_PENDING)
        {
                return FALSE;
        }

        return GetOverlappedResult(hi,pO,&bytes,TRUE);
}
#define CHECK_IO(h,o)  if(!wait_for_io(h,o)) {goto done;}

int is_gui(char *exename) {

        HANDLE hImage;

        DWORD  bytes;
        OVERLAPPED overlap;

        ULONG  ntSignature;

        struct DosHeader{
                IMAGE_DOS_HEADER     doshdr;
                DWORD                extra[16];
        };

        struct DosHeader dh;
        IMAGE_OPTIONAL_HEADER optionalhdr;

        int retCode = 0;

        memset(&overlap,0,sizeof(overlap));


        hImage = CreateFile(exename, GENERIC_READ, FILE_SHARE_READ, NULL,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL| FILE_FLAG_OVERLAPPED, NULL);
        if (INVALID_HANDLE_VALUE == hImage) {
                return 0;
        }

        (void)ReadFile(hImage, &dh, sizeof(struct DosHeader), &bytes,&overlap);
        CHECK_IO(hImage,&overlap);


        if (IMAGE_DOS_SIGNATURE != dh.doshdr.e_magic) {
                goto done;
        }

        // read from the coffheaderoffset;
        overlap.Offset = dh.doshdr.e_lfanew;

        (void)ReadFile(hImage, &ntSignature, sizeof(ULONG), &bytes,&overlap);
        CHECK_IO(hImage,&overlap);

        if (IMAGE_NT_SIGNATURE != ntSignature) {
                goto done;
        }
        overlap.Offset = dh.doshdr.e_lfanew + sizeof(ULONG) +
                sizeof(IMAGE_FILE_HEADER);

        (void)ReadFile(hImage, &optionalhdr,IMAGE_SIZEOF_NT_OPTIONAL_HEADER, &bytes,&overlap);
        CHECK_IO(hImage,&overlap);

        if (optionalhdr.Subsystem ==IMAGE_SUBSYSTEM_WINDOWS_GUI)
                retCode =  1;
done:
        CloseHandle(hImage);
        return retCode;
}
int is_9x_gui(char *prog) {
	
	char *progpath;
	DWORD dwret;
	char *pathbuf;
	char *pext;
	
	pathbuf=heap_alloc(MAX_PATH+1);
	if(!pathbuf)
		return 0;

	progpath=heap_alloc((MAX_PATH<<1)+1);
	if(!progpath)
		return 0;

	if (GetEnvironmentVariable("PATH",pathbuf,MAX_PATH) ==0) {
		goto failed;
	}
	
	pathbuf[MAX_PATH]=0;

	dwret = SearchPath(pathbuf,prog,".EXE",MAX_PATH<<1,progpath,&pext);

	if ( (dwret == 0) || (dwret > (MAX_PATH<<1) ) )
		goto failed;
	
	dprintf("progpath is %s\n",progpath);
	dwret = is_gui(progpath);

	heap_free(pathbuf);
	heap_free(progpath);

	return dwret;

failed:
	heap_free(pathbuf);
	heap_free(progpath);
	return 0;


}
