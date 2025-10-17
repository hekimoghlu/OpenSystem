/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 17, 2023.
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
#ifdef _WIN32

#include <ctype.h>
#include <io.h>
#include <wchar.h>
#include <windows.h>

#ifdef USE_FILEEXTD
// VC 7.1 or earlier doesn't support SAL.
# if !defined(_MSC_VER) || (_MSC_VER < 1400)
#  define __out
#  define __in
#  define __in_opt
# endif
// Win32 FileID API Library:
// http://www.microsoft.com/en-us/download/details.aspx?id=22599
// Needed for WinXP.
# include <fileextd.h>
#else // USE_FILEEXTD
// VC 8 or earlier.
# if defined(_MSC_VER) && (_MSC_VER < 1500)
#  ifdef ENABLE_STUB_IMPL
#   define STUB_IMPL
#  else
#   error "Win32 FileID API Library is required for VC2005 or earlier."
#  endif
# endif
#endif // USE_FILEEXTD

#ifdef __MINGW32__
# define UNUSED __attribute__((unused))
#else
# define UNUSED
#endif

#include "iscygpty.h"

//#define USE_DYNFILEID
#ifdef USE_DYNFILEID
typedef BOOL (WINAPI *pfnGetFileInformationByHandleEx)(
		HANDLE						hFile,
		FILE_INFO_BY_HANDLE_CLASS	FileInformationClass,
		LPVOID						lpFileInformation,
		DWORD						dwBufferSize);
static pfnGetFileInformationByHandleEx pGetFileInformationByHandleEx = NULL;

# ifndef USE_FILEEXTD
static BOOL WINAPI stub_GetFileInformationByHandleEx(
		HANDLE						hFile UNUSED,
		FILE_INFO_BY_HANDLE_CLASS	FileInformationClass UNUSED,
		LPVOID						lpFileInformation UNUSED,
		DWORD						dwBufferSize UNUSED)
{
	return FALSE;
}
# endif

static void setup_fileid_api(void)
{
	if (pGetFileInformationByHandleEx != NULL) {
		return;
	}
	pGetFileInformationByHandleEx = (pfnGetFileInformationByHandleEx)
		GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")),
				"GetFileInformationByHandleEx");
	if (pGetFileInformationByHandleEx == NULL) {
# ifdef USE_FILEEXTD
		pGetFileInformationByHandleEx = GetFileInformationByHandleEx;
# else
		pGetFileInformationByHandleEx = stub_GetFileInformationByHandleEx;
# endif
	}
}
#else
# define pGetFileInformationByHandleEx	GetFileInformationByHandleEx
# define setup_fileid_api()
#endif


#define is_wprefix(s, prefix) \
	(wcsncmp((s), (prefix), sizeof(prefix) / sizeof(WCHAR) - 1) == 0)

// Check if the fd is a cygwin/msys's pty.
int is_cygpty(int fd)
{
#ifdef STUB_IMPL
	return 0;
#else
	HANDLE h;
	const int size = sizeof(FILE_NAME_INFO) + sizeof(WCHAR) * (MAX_PATH - 1);
	FILE_NAME_INFO *nameinfo;
	WCHAR *p = NULL;

	setup_fileid_api();

	h = (HANDLE) _get_osfhandle(fd);
	if (h == INVALID_HANDLE_VALUE) {
		return 0;
	}
	// Cygwin/msys's pty is a pipe.
	if (GetFileType(h) != FILE_TYPE_PIPE) {
		return 0;
	}
	nameinfo = malloc(size + sizeof(WCHAR));
	if (nameinfo == NULL) {
		return 0;
	}
	// Check the name of the pipe:
	// "\\{cygwin,msys}-XXXXXXXXXXXXXXXX-ptyN-{from,to}-master"
	if (pGetFileInformationByHandleEx(h, FileNameInfo, nameinfo, size)) {
		nameinfo->FileName[nameinfo->FileNameLength / sizeof(WCHAR)] = L'\0';
		p = nameinfo->FileName;
		if (is_wprefix(p, L"\\cygwin-")) {		// Cygwin
			p += 8;
		} else if (is_wprefix(p, L"\\msys-")) {	// MSYS and MSYS2
			p += 6;
		} else {
			p = NULL;
		}
		if (p != NULL) {
			// Skip 16-digit hexadecimal.
			while (*p && iswascii(*p) && isxdigit(*p))
				++p;
			if (is_wprefix(p, L"-pty")) {
				p += 4;
			} else {
				p = NULL;
			}
		}
		if (p != NULL) {
			// Skip pty number.
			while (*p && iswascii(*p) && isdigit(*p))
				++p;
			if (is_wprefix(p, L"-from-master")) {
				//p += 12;
			} else if (is_wprefix(p, L"-to-master")) {
				//p += 10;
			} else {
				p = NULL;
			}
		}
	}
	free(nameinfo);
	return (p != NULL);
#endif // STUB_IMPL
}

// Check if at least one cygwin/msys pty is used.
int is_cygpty_used(void)
{
	int fd, ret = 0;

	for (fd = 0; fd < 3; fd++) {
		ret |= is_cygpty(fd);
	}
	return ret;
}

#endif // _WIN32

// vim: set ts=4 sw=4:
