/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#include "kx.h"

RCSID("$Id$");

/*
 * Set the common part of the context `kc'
 */

void
context_set (kx_context *kc, const char *host, const char *user, int port,
	     int debug_flag, int keepalive_flag, int tcp_flag)
{
    kc->thisaddr        = (struct sockaddr*)&kc->__ss_this;
    kc->thataddr        = (struct sockaddr*)&kc->__ss_that;
    kc->host		= host;
    kc->user		= user;
    kc->port		= port;
    kc->debug_flag	= debug_flag;
    kc->keepalive_flag	= keepalive_flag;
    kc->tcp_flag	= tcp_flag;
}

/*
 * dispatch functions
 */

void
context_destroy (kx_context *kc)
{
    (*kc->destroy)(kc);
}

int
context_authenticate (kx_context *kc, int s)
{
    return (*kc->authenticate)(kc, s);
}

int
context_userok (kx_context *kc, char *user)
{
    return (*kc->userok)(kc, user);
}

ssize_t
kx_read (kx_context *kc, int fd, void *buf, size_t len)
{
    return (*kc->read)(kc, fd, buf, len);
}

ssize_t
kx_write (kx_context *kc, int fd, const void *buf, size_t len)
{
    return (*kc->write)(kc, fd, buf, len);
}

int
copy_encrypted (kx_context *kc, int fd1, int fd2)
{
    return (*kc->copy_encrypted)(kc, fd1, fd2);
}
