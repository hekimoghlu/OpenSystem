/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#include "FEATURE/uwin"

#if !_UWIN || _lib_getpass

void _STUB_getpass(){}

#else

#pragma prototyped

#define getpass	______getpass

#include	<ast.h>
#include	<termios.h>
#include	<signal.h>

#undef	getpass

#if defined(__EXPORT__)
#define extern		__EXPORT__
#endif

static int interrupt;
static void handler(int sig)
{
	interrupt++;
}

extern char*	getpass(const char *prompt)
{
	struct termios told,tnew;
	Sfio_t *iop;
	static char *cp, passwd[32];
	void (*savesig)(int);
	if(!(iop = sfopen((Sfio_t*)0, "/dev/tty", "r")))
		return(0);
	if(tcgetattr(sffileno(iop),&told) < 0)
		return(0);
	interrupt = 0;
	tnew = told;
	tnew.c_lflag &= ~(ECHO|ECHOE|ECHOK|ECHONL);
	if(tcsetattr(sffileno(iop),TCSANOW,&tnew) < 0)
		return(0);
	savesig = signal(SIGINT, handler);
	sfputr(sfstderr,prompt,-1);
	if(cp = sfgetr(iop,'\n',1))
		strncpy(passwd,cp,sizeof(passwd)-1);
	tcsetattr(sffileno(iop),TCSANOW,&told);
	sfputc(sfstderr,'\n');
	sfclose(iop);
	signal(SIGINT, savesig);
	if(interrupt)
		kill(getpid(),SIGINT);
	return(cp?passwd:0);
}


#endif
