/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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
#include <stdio.h>
#include "expect.h"

timedout()
{
	fprintf(stderr,"timed out\n");
	exit(-1);
}

char move[100];

read_first_move(fd)
int fd;
{
	if (EXP_TIMEOUT == exp_expectl(fd,
			exp_glob,"first\r\n1.*\r\n",0,
			exp_end)) {
		timedout();
	}
	sscanf(exp_match,"%*s 1. %s",move);
}

/* moves and counter-moves are printed out in different formats, sigh... */

read_counter_move(fd)
int fd;
{
	switch (exp_expectl(fd,exp_glob,"*...*\r\n",0,exp_end)) {
	case EXP_TIMEOUT: timedout();
	case EXP_EOF: exit(-1);
	}

	sscanf(exp_match,"%*s %*s %*s %*s ... %s",move);
}

read_move(fd)
int fd;
{
	switch (exp_expectl(fd,exp_glob,"*...*\r\n*.*\r\n",0,exp_end)) {
	case EXP_TIMEOUT: timedout();
	case EXP_EOF: exit(-1);
	}

	sscanf(exp_match,"%*s %*s ... %*s %*s %s",move);
}

send_move(fd)
int fd;
{
	write(fd,move,strlen(move));
}

main(){
	int fd1, fd2;

	exp_loguser = 1;
	exp_timeout = 3600;

	if (-1 == (fd1 = exp_spawnl("chess","chess",(char *)0))) {
	  perror("chess");
	  exit(-1);
	}

	if (-1 == exp_expectl(fd1,exp_glob,"Chess\r\n",0,exp_end)) exit;

	if (-1 == write(fd1,"first\r",6)) exit;

	read_first_move(fd1);

	fd2 = exp_spawnl("chess","chess",(char *)0);

	if (-1 == exp_expectl(fd2,exp_glob,"Chess\r\n",0,exp_end)) exit;

	for (;;) {
		send_move(fd2);
		read_counter_move(fd2);

		send_move(fd1);
		read_move(fd1);
	}
}
