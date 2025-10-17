/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#include "rubysocket.h"

#ifdef SOCKS
/*
 * call-seq:
 *   SOCKSSocket.new(host, serv) => socket
 *
 * Opens a SOCKS connection to +host+ via the SOCKS server +serv+.
 *
 */
static VALUE
socks_init(VALUE sock, VALUE host, VALUE serv)
{
    static int init = 0;

    if (init == 0) {
	SOCKSinit("ruby");
	init = 1;
    }

    return rsock_init_inetsock(sock, host, serv, Qnil, Qnil, INET_SOCKS);
}

#ifdef SOCKS5
/*
 * Closes the SOCKS connection.
 *
 */
static VALUE
socks_s_close(VALUE sock)
{
    rb_io_t *fptr;

    GetOpenFile(sock, fptr);
    shutdown(fptr->fd, 2);
    return rb_io_close(sock);
}
#endif
#endif

void
rsock_init_sockssocket(void)
{
#ifdef SOCKS
    /*
     * Document-class: SOCKSSocket < TCPSocket
     *
     * SOCKS is an Internet protocol that routes packets between a client and
     * a server through a proxy server.  SOCKS5, if supported, additionally
     * provides authentication so only authorized users may access a server.
     */
    rb_cSOCKSSocket = rb_define_class("SOCKSSocket", rb_cTCPSocket);
    rb_define_method(rb_cSOCKSSocket, "initialize", socks_init, 2);
#ifdef SOCKS5
    rb_define_method(rb_cSOCKSSocket, "close", socks_s_close, 0);
#endif
#endif
}
