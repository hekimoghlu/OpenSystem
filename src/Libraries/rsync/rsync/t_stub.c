/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
#include "rsync.h"

int modify_window = 0;
int module_id = -1;
int relative_paths = 0;
int human_readable = 0;
int module_dirlen = 0;
mode_t orig_umask = 002;
char *partial_dir;
struct filter_list_struct server_filter_list;

 void rprintf(UNUSED(enum logcode code), const char *format, ...)
{
	va_list ap;
	va_start(ap, format);
	vfprintf(stderr, format, ap);
	va_end(ap);
}

 void rsyserr(UNUSED(enum logcode code), int errcode, const char *format, ...)
{
	va_list ap;
	fputs(RSYNC_NAME ": ", stderr);
	va_start(ap, format);
	vfprintf(stderr, format, ap);
	va_end(ap);
	fprintf(stderr, ": %s (%d)\n", strerror(errcode), errcode);
}

 void _exit_cleanup(int code, const char *file, int line)
{
	fprintf(stderr, "exit(%d): %s(%d)\n",
		code, file, line);
	exit(code);
}

 int check_filter(UNUSED(struct filter_list_struct *listp), UNUSED(char *name),
		   UNUSED(int name_is_dir))
{
	/* This function doesn't really get called in this test context, so
	 * just return 0. */
	return 0;
}

 char *lp_name(UNUSED(int mod))
{
	return NULL;
}

 BOOL lp_use_chroot(UNUSED(int mod))
{
	return 0;
}

 char *lp_path(UNUSED(int mod))
{
	return NULL;
}

 const char *who_am_i(void)
{
	return "tester";
}
