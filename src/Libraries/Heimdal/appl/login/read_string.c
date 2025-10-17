/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
#include "login_locl.h"

RCSID("$Id$");

static sig_atomic_t intr_flag;

static void
intr(int sig)
{
    intr_flag++;
}

#ifndef NSIG
#define NSIG 47
#endif

int
read_string(const char *prompt, char *buf, size_t len, int echo)
{
    struct sigaction sigs[NSIG];
    int oksigs[NSIG];
    struct sigaction sa;
    FILE *tty;
    int ret = 0;
    int of = 0;
    int i;
    int c;
    char *p;

    struct termios t_new, t_old;

    memset(&oksigs, 0, sizeof(oksigs));

    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = intr;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    for(i = 1; i < sizeof(sigs) / sizeof(sigs[0]); i++)
	if (i != SIGALRM)
	    if (sigaction(i, &sa, &sigs[i]) == 0)
		oksigs[i] = 1;

    if((tty = fopen("/dev/tty", "r")) == NULL)
	tty = stdin;

    fprintf(stderr, "%s", prompt);
    fflush(stderr);

    if(echo == 0){
	tcgetattr(fileno(tty), &t_old);
	memcpy(&t_new, &t_old, sizeof(t_new));
	t_new.c_lflag &= ~ECHO;
	tcsetattr(fileno(tty), TCSANOW, &t_new);
    }
    intr_flag = 0;
    p = buf;
    while(intr_flag == 0){
	c = getc(tty);
	if(c == EOF){
	    if(!ferror(tty))
		ret = 1;
	    break;
	}
	if(c == '\n')
	    break;
	if(of == 0)
	    *p++ = c;
	of = (p == buf + len);
    }
    if(of)
	p--;
    *p = 0;

    if(echo == 0){
	printf("\n");
	tcsetattr(fileno(tty), TCSANOW, &t_old);
    }

    if(tty != stdin)
	fclose(tty);

    for(i = 1; i < sizeof(sigs) / sizeof(sigs[0]); i++)
	if (oksigs[i])
	    sigaction(i, &sigs[i], NULL);

    if(ret)
	return -3;
    if(intr_flag)
	return -2;
    if(of)
	return -1;
    return 0;
}


#if 0
int main()
{
    char s[128];
    int ret;
    ret = read_string("foo: ", s, sizeof(s), 0);
    printf("%d ->%s<-\n", ret, s);
}
#endif
