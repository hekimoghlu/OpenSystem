/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
 * bogus.c: various routines that are really silly
 * -amol
 *
 */
#include "ntport.h"
#include "sh.h"

static struct passwd pass_bogus;
static char username[20];
static char homedir[MAX_PATH + 1];/*FIXBUF*/
static char *this_shell="tcsh";

static char dummy[2]={0,0};

gid_t getuid(void) {
	return 0;
}
gid_t getgid(void) {
	return 0;
}
gid_t geteuid(void) {
	return 0;
}
gid_t getegid(void) {
	return 0;
}
#undef free
struct passwd * getpwnam(const char *name) {

	char *ptr;
	DWORD size =20;
	size_t esize = 0;

	if (pass_bogus.pw_name == NULL) {
		GetUserName(username,&size);
		if (_dupenv_s(&ptr,&esize,"HOME") == 0){
			StringCbCopy(homedir,sizeof(homedir),ptr);
			pass_bogus.pw_dir = &homedir[0];
			free(ptr);
		}
		pass_bogus.pw_name = &username[0];
		pass_bogus.pw_shell = this_shell;


		pass_bogus.pw_passwd= &dummy[0];
		pass_bogus.pw_gecos=&dummy[0];
		pass_bogus.pw_passwd= &dummy[0];

	}
	if (_stricmp(username,name) )
		return NULL;
	return &pass_bogus;
}
struct passwd * getpwuid(uid_t myuid) {

	char *ptr;
	DWORD size =20;
	size_t esize = 0;

	UNREFERENCED_PARAMETER(myuid);
	if (pass_bogus.pw_name == NULL) {
		GetUserName(username,&size);
		if (_dupenv_s(&ptr,&esize,"HOME") == 0){
			StringCbCopy(homedir,sizeof(homedir),ptr);
			pass_bogus.pw_dir = &homedir[0];
			free(ptr);
		}
		pass_bogus.pw_name = &username[0];
		pass_bogus.pw_shell = this_shell;


		pass_bogus.pw_passwd= &dummy[0];
		pass_bogus.pw_gecos=&dummy[0];
		pass_bogus.pw_passwd= &dummy[0];

	}
	return &pass_bogus;
}
struct group * getgrnam(char *name) {
	UNREFERENCED_PARAMETER(name);
	return NULL;
}
struct group * getgrgid(gid_t mygid) {
	UNREFERENCED_PARAMETER(mygid);
	return NULL;
}
char * ttyname(int fd) {

	if (isatty(fd)) return "/dev/tty";
	return NULL;
}
int times(struct tms * ignore) {
	FILETIME c,e,kernel,user;

	ignore->tms_utime=0;
	ignore->tms_stime=0;
	ignore->tms_cutime=0;
	ignore->tms_cstime=0;
	if (!GetProcessTimes(GetCurrentProcess(),
				&c,
				&e,
				&kernel,
				&user) )
		return -1;

	if (kernel.dwHighDateTime){
		return GetTickCount();
	}
	//
	// Units of 10ms. I *think* this is right. -amol 6/2/97
	ignore->tms_stime = kernel.dwLowDateTime / 1000 /100;
	ignore->tms_utime = user.dwLowDateTime / 1000 /100;

	return GetTickCount();
}
int tty_getty(int fd, void*ignore) {
	UNREFERENCED_PARAMETER(fd);
	UNREFERENCED_PARAMETER(ignore);
	return 0;
}
int tty_setty(int fd, void*ignore) {
	UNREFERENCED_PARAMETER(fd);
	UNREFERENCED_PARAMETER(ignore);
	return 0;
}
int tty_geteightbit(void *ignore) {
	UNREFERENCED_PARAMETER(ignore);
	return 1;
}
	void
dosetty(Char **v, struct command *t)
{
	UNREFERENCED_PARAMETER(v);
	UNREFERENCED_PARAMETER(t);
	xprintf("setty not supported in NT\n");
}

