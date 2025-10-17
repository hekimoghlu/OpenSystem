/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#include "apr_network_io.h"
#include "apr_poll.h"

APR_DECLARE(apr_status_t) apr_socket_atreadeof(apr_socket_t *sock, int *atreadeof)
{
    apr_pollfd_t pfds[1];
    apr_status_t rv;
    apr_int32_t  nfds;

    /* The purpose here is to return APR_SUCCESS only in cases in
     * which it can be unambiguously determined whether or not the
     * socket will return EOF on next read.  In case of an unexpected
     * error, return that. */

    pfds[0].reqevents = APR_POLLIN;
    pfds[0].desc_type = APR_POLL_SOCKET;
    pfds[0].desc.s = sock;

    do {
        rv = apr_poll(&pfds[0], 1, &nfds, 0);
    } while (APR_STATUS_IS_EINTR(rv));

    if (APR_STATUS_IS_TIMEUP(rv)) {
        /* Read buffer empty -> subsequent reads would block, so,
         * definitely not at EOF. */
        *atreadeof = 0;
        return APR_SUCCESS;
    }
    else if (rv) {
        /* Some other error -> unexpected error. */
        return rv;
    }
    /* Many platforms return only APR_POLLIN; OS X returns APR_POLLHUP|APR_POLLIN */
    else if (nfds == 1 && (pfds[0].rtnevents & APR_POLLIN)  == APR_POLLIN) {
        apr_sockaddr_t unused;
        apr_size_t len = 1;
        char buf;

        /* The socket is readable - peek to see whether it returns EOF
         * without consuming bytes from the socket buffer. */
        rv = apr_socket_recvfrom(&unused, sock, MSG_PEEK, &buf, &len);
        if (rv == APR_EOF) {
            *atreadeof = 1;
            return APR_SUCCESS;
        }
        else if (rv) {
            /* Read error -> unexpected error. */
            return rv;
        }
        else {
            *atreadeof = 0;
            return APR_SUCCESS;
        }
    }

    /* Should not fall through here. */
    return APR_EGENERAL;
}

