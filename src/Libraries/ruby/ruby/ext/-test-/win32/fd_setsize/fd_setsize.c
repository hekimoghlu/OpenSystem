/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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

#undef FD_SETSIZE
/* redefine smaller size then default 64 */
#define FD_SETSIZE 32
#include <ruby.h>

static VALUE
test_select(VALUE self)
{
    int sd = socket(AF_INET, SOCK_DGRAM, 0);
    struct timeval zero;
    fd_set read;
    fd_set write;
    fd_set error;

    zero.tv_sec = 0;
    zero.tv_usec = 0;

    FD_ZERO(&read);
    FD_ZERO(&write);
    FD_ZERO(&error);

    FD_SET(sd, &read);
    FD_SET(sd, &write);
    FD_SET(sd, &error);

    select(sd+1, &read, &write, &error, &zero);

    return Qtrue;
}

static VALUE
test_fdset(VALUE self)
{
    int i;
    fd_set set;

    FD_ZERO(&set);

    for (i = 0; i < FD_SETSIZE * 2; i++) {
	int sd = socket(AF_INET, SOCK_DGRAM, 0);
	FD_SET(sd, &set);
	if (set.fd_count > FD_SETSIZE) {
	    return Qfalse;
	}
    }
    return Qtrue;
}

void
Init_fd_setsize(void)
{
    VALUE m = rb_define_module_under(rb_define_module("Bug"), "Win32");
    rb_define_module_function(m, "test_select", test_select, 0);
    rb_define_module_function(m, "test_fdset", test_fdset, 0);
}
