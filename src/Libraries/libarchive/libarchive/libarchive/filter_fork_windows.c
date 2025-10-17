/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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
#include "archive_platform.h"

#if defined(_WIN32) && !defined(__CYGWIN__)
#include "archive_cmdline_private.h"
#include "archive_string.h"

#include "filter_fork.h"

#if !defined(WINAPI_FAMILY_PARTITION) || WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
/* There are some editions of Windows ("nano server," for example) that
 * do not host user32.dll. If we want to keep running on those editions,
 * we need to delay-load WaitForInputIdle. */
static void *
la_GetFunctionUser32(const char *name)
{
	static HINSTANCE lib;
	static int set;
	if (!set) {
		set = 1;
		lib = LoadLibrary(TEXT("user32.dll"));
	}
	if (lib == NULL) {
		return NULL;
	}
	return (void *)GetProcAddress(lib, name);
}

static int
la_WaitForInputIdle(HANDLE hProcess, DWORD dwMilliseconds)
{
	static DWORD (WINAPI *f)(HANDLE, DWORD);
	static int set;

	if (!set) {
		set = 1;
		f = la_GetFunctionUser32("WaitForInputIdle");
	}

	if (!f) {
		/* An inability to wait for input idle is
		 * not _good_, but it is not catastrophic. */
		return WAIT_FAILED;
	}
	return (*f)(hProcess, dwMilliseconds);
}

int
__archive_create_child(const char *cmd, int *child_stdin, int *child_stdout,
		HANDLE *out_child)
{
	HANDLE childStdout[2], childStdin[2],childStderr;
	SECURITY_ATTRIBUTES secAtts;
	STARTUPINFOA staInfo;
	PROCESS_INFORMATION childInfo;
	struct archive_string cmdline;
	struct archive_string fullpath;
	struct archive_cmdline *acmd;
	char *arg0, *ext;
	int i, l;
	DWORD fl, fl_old;
	HANDLE child;

	childStdout[0] = childStdout[1] = INVALID_HANDLE_VALUE;
	childStdin[0] = childStdin[1] = INVALID_HANDLE_VALUE;
	childStderr = INVALID_HANDLE_VALUE;
	archive_string_init(&cmdline);
	archive_string_init(&fullpath);

	acmd = __archive_cmdline_allocate();
	if (acmd == NULL)
		goto fail;
	if (__archive_cmdline_parse(acmd, cmd) != ARCHIVE_OK)
		goto fail;

	/*
	 * Search the full path of 'path'.
	 * NOTE: This does not need if we give CreateProcessA 'path' as
	 * a part of the cmdline and give CreateProcessA NULL as first
	 * parameter, but I do not like that way.
	 */
	ext = strrchr(acmd->path, '.');
	if (ext == NULL || strlen(ext) > 4)
		/* 'path' does not have a proper extension, so we have to
		 * give SearchPath() ".exe" as the extension. */
		ext = ".exe";
	else
		ext = NULL;/* 'path' has an extension. */

	fl = MAX_PATH;
	do {
		if (archive_string_ensure(&fullpath, fl) == NULL)
			goto fail;
		fl_old = fl;
		fl = SearchPathA(NULL, acmd->path, ext, fl, fullpath.s,
			&arg0);
	} while (fl != 0 && fl > fl_old);
	if (fl == 0)
		goto fail;

	/*
	 * Make a command line.
	 */
	for (l = 0, i = 0;  acmd->argv[i] != NULL; i++) {
		if (i == 0)
			continue;
		l += (int)strlen(acmd->argv[i]) + 1;
	}
	if (archive_string_ensure(&cmdline, l + 1) == NULL)
		goto fail;
	for (i = 0;  acmd->argv[i] != NULL; i++) {
		if (i == 0) {
			const char *p, *sp;

			if ((p = strchr(acmd->argv[i], '/')) != NULL ||
			    (p = strchr(acmd->argv[i], '\\')) != NULL)
				p++;
			else
				p = acmd->argv[i];
			if ((sp = strchr(p, ' ')) != NULL)
				archive_strappend_char(&cmdline, '"');
			archive_strcat(&cmdline, p);
			if (sp != NULL)
				archive_strappend_char(&cmdline, '"');
		} else {
			archive_strappend_char(&cmdline, ' ');
			archive_strcat(&cmdline, acmd->argv[i]);
		}
	}
	if (i <= 1) {
		const char *sp;

		if ((sp = strchr(arg0, ' ')) != NULL)
			archive_strappend_char(&cmdline, '"');
		archive_strcat(&cmdline, arg0);
		if (sp != NULL)
			archive_strappend_char(&cmdline, '"');
	}

	secAtts.nLength = sizeof(SECURITY_ATTRIBUTES);
	secAtts.bInheritHandle = TRUE;
	secAtts.lpSecurityDescriptor = NULL;
	if (CreatePipe(&childStdout[0], &childStdout[1], &secAtts, 0) == 0)
		goto fail;
	if (!SetHandleInformation(childStdout[0], HANDLE_FLAG_INHERIT, 0))
		goto fail;
	if (CreatePipe(&childStdin[0], &childStdin[1], &secAtts, 0) == 0)
		goto fail;
	if (!SetHandleInformation(childStdin[1], HANDLE_FLAG_INHERIT, 0))
		goto fail;
	if (DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE),
	    GetCurrentProcess(), &childStderr, 0, TRUE,
	    DUPLICATE_SAME_ACCESS) == 0)
		goto fail;

	memset(&staInfo, 0, sizeof(staInfo));
	staInfo.cb = sizeof(staInfo);
	staInfo.hStdError = childStderr;
	staInfo.hStdOutput = childStdout[1];
	staInfo.hStdInput = childStdin[0];
	staInfo.wShowWindow = SW_HIDE;
	staInfo.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
	if (CreateProcessA(fullpath.s, cmdline.s, NULL, NULL, TRUE, 0,
	      NULL, NULL, &staInfo, &childInfo) == 0)
		goto fail;
	la_WaitForInputIdle(childInfo.hProcess, INFINITE);
	CloseHandle(childInfo.hProcess);
	CloseHandle(childInfo.hThread);

	*child_stdout = _open_osfhandle((intptr_t)childStdout[0], _O_RDONLY);
	*child_stdin = _open_osfhandle((intptr_t)childStdin[1], _O_WRONLY);
	
	child = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE,
		childInfo.dwProcessId);
	if (child == NULL) // INVALID_HANDLE_VALUE ?
		goto fail;

	*out_child = child;

	CloseHandle(childStdout[1]);
	CloseHandle(childStdin[0]);

	archive_string_free(&cmdline);
	archive_string_free(&fullpath);
	__archive_cmdline_free(acmd);
	return ARCHIVE_OK;

fail:
	if (childStdout[0] != INVALID_HANDLE_VALUE)
		CloseHandle(childStdout[0]);
	if (childStdout[1] != INVALID_HANDLE_VALUE)
		CloseHandle(childStdout[1]);
	if (childStdin[0] != INVALID_HANDLE_VALUE)
		CloseHandle(childStdin[0]);
	if (childStdin[1] != INVALID_HANDLE_VALUE)
		CloseHandle(childStdin[1]);
	if (childStderr != INVALID_HANDLE_VALUE)
		CloseHandle(childStderr);
	archive_string_free(&cmdline);
	archive_string_free(&fullpath);
	__archive_cmdline_free(acmd);
	return ARCHIVE_FAILED;
}
#else /* !WINAPI_PARTITION_DESKTOP */
int
__archive_create_child(const char *cmd, int *child_stdin, int *child_stdout, HANDLE *out_child)
{
	(void)cmd; (void)child_stdin; (void) child_stdout; (void) out_child;
	return ARCHIVE_FAILED;
}
#endif /* !WINAPI_PARTITION_DESKTOP */

void
__archive_check_child(int in, int out)
{
	(void)in; /* UNUSED */
	(void)out; /* UNUSED */
	Sleep(100);
}

#endif /* _WIN32 && !__CYGWIN__ */
