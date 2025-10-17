/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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

void Init_golf(void);
#define ruby_options goruby_options
#define ruby_run_node goruby_run_node
#include "main.c"
#undef ruby_options
#undef ruby_run_node

#if defined _WIN32
#include <io.h>
#include <fcntl.h>
#define pipe(p) _pipe(p, 32L, _O_NOINHERIT)
#elif defined HAVE_UNISTD_H
#include <unistd.h>
#endif

RUBY_EXTERN void *ruby_options(int argc, char **argv);
RUBY_EXTERN int ruby_run_node(void*);
RUBY_EXTERN void ruby_init_ext(const char *name, void (*init)(void));

static VALUE
init_golf(VALUE arg)
{
    Init_golf();
    rb_provide("golf.so");
    return arg;
}

void *
goruby_options(int argc, char **argv)
{
    static const char cmd[] = "END{require 'irb';IRB.start}";
    int rw[2], infd;
    void *ret;

    if ((isatty(0) && isatty(1) && isatty(2)) && (pipe(rw) == 0)) {
	ssize_t n;
	infd = dup(0);
	if (infd < 0) {
	    close(rw[0]);
	    close(rw[1]);
	    goto no_irb;
	}
	dup2(rw[0], 0);
	close(rw[0]);
	n = write(rw[1], cmd, sizeof(cmd) - 1);
	close(rw[1]);
	ret = n > 0 ? ruby_options(argc, argv) : NULL;
	dup2(infd, 0);
	close(infd);
	return ret;
    }
    else {
      no_irb:
	return ruby_options(argc, argv);
    }
}

int
goruby_run_node(void *arg)
{
    int state;
    if (NIL_P(rb_protect(init_golf, Qtrue, &state))) {
	return state == EXIT_SUCCESS ? EXIT_FAILURE : state;
    }
    return ruby_run_node(arg);
}
