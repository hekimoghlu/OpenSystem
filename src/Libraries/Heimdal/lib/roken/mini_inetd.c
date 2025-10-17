/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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
#include <config.h>

#include <err.h>
#include "roken.h"

/*
 * accept a connection on `s' and pretend it's served by inetd.
 */

static void
accept_it (rk_socket_t s, rk_socket_t *ret_socket)
{
    rk_socket_t as;

    as = accept(s, NULL, NULL);
    if(rk_IS_BAD_SOCKET(as))
	err (1, "accept");

    if (ret_socket) {

	*ret_socket = as;

    } else {
	int fd = socket_to_fd(as, 0);

	/* We would use _O_RDONLY for the socket_to_fd() call for
	   STDIN, but there are instances where we assume that STDIN
	   is a r/w socket. */

	dup2(fd, STDIN_FILENO);
	dup2(fd, STDOUT_FILENO);

	rk_closesocket(as);
    }
}

/**
 * Listen on a specified addresses
 *
 * Listens on the specified addresses for incoming connections.  If
 * the \a ret_socket parameter is \a NULL, on return STDIN and STDOUT
 * will be connected to an accepted socket.  If the \a ret_socket
 * parameter is non-NULL, the accepted socket will be returned in
 * *ret_socket.  In the latter case, STDIN and STDOUT will be left
 * unmodified.
 *
 * This function does not return if there is an error or if no
 * connection is established.
 *
 * @param[in] ai Addresses to listen on
 * @param[out] ret_socket If non-NULL receives the accepted socket.
 *
 * @see mini_inetd()
 */
ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
mini_inetd_addrinfo (struct addrinfo *ai, rk_socket_t *ret_socket)
{
    int ret;
    struct addrinfo *a;
    int n, nalloc, i;
    rk_socket_t *fds;
    fd_set orig_read_set, read_set;
    rk_socket_t max_fd = (rk_socket_t)-1;

    for (nalloc = 0, a = ai; a != NULL; a = a->ai_next)
	++nalloc;
    
    if (nalloc == 0) {
	errx(1, "mini_inetd: no sockets to listen on");
	UNREACHABLE(return);
    }

    fds = malloc (nalloc * sizeof(*fds));
    if (fds == NULL) {
	errx (1, "mini_inetd: out of memory");
	UNREACHABLE(return);
    }

    FD_ZERO(&orig_read_set);

    for (i = 0, a = ai; a != NULL; a = a->ai_next) {
	fds[i] = socket (a->ai_family, a->ai_socktype, a->ai_protocol);
	if (rk_IS_BAD_SOCKET(fds[i]))
	    continue;
	socket_set_nopipe(fds[i], 1);
	socket_set_reuseaddr (fds[i], 1);
	socket_set_ipv6only(fds[i], 1);
	if (rk_IS_SOCKET_ERROR(bind (fds[i], a->ai_addr, a->ai_addrlen))) {
	    warn ("bind af = %d", a->ai_family);
	    rk_closesocket(fds[i]);
	    fds[i] = rk_INVALID_SOCKET;
	    continue;
	}
	if (rk_IS_SOCKET_ERROR(listen (fds[i], SOMAXCONN))) {
	    warn ("listen af = %d", a->ai_family);
	    rk_closesocket(fds[i]);
	    fds[i] = rk_INVALID_SOCKET;
	    continue;
	}
#ifndef NO_LIMIT_FD_SETSIZE
	if (fds[i] >= FD_SETSIZE)
	    errx (1, "fd too large");
#endif
	FD_SET(fds[i], &orig_read_set);
	max_fd = max(max_fd, fds[i]);
	++i;
    }
    if (i == 0)
	errx (1, "no sockets");
    n = i;

    do {
	read_set = orig_read_set;

	ret = select (max_fd + 1, &read_set, NULL, NULL, NULL);
	if (rk_IS_SOCKET_ERROR(ret) && rk_SOCK_ERRNO != EINTR)
	    err (1, "select");
    } while (ret <= 0);

    for (i = 0; i < n; ++i)
	if (FD_ISSET (fds[i], &read_set)) {
	    accept_it (fds[i], ret_socket);
	    for (i = 0; i < n; ++i)
	      rk_closesocket(fds[i]);
	    free(fds);
	    return;
	}
    abort ();
}

/**
 * Listen on a specified port
 *
 * Listens on the specified port for incoming connections.  If the \a
 * ret_socket parameter is \a NULL, on return STDIN and STDOUT will be
 * connected to an accepted socket.  If the \a ret_socket parameter is
 * non-NULL, the accepted socket will be returned in *ret_socket.  In
 * the latter case, STDIN and STDOUT will be left unmodified.
 *
 * This function does not return if there is an error or if no
 * connection is established.
 *
 * @param[in] port Port to listen on
 * @param[out] ret_socket If non-NULL receives the accepted socket.
 *
 * @see mini_inetd_addrinfo()
 */
ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
mini_inetd(int port, rk_socket_t * ret_socket)
{
    int error;
    struct addrinfo *ai, hints;
    char portstr[NI_MAXSERV];

    memset (&hints, 0, sizeof(hints));
    hints.ai_flags    = AI_PASSIVE;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family   = PF_UNSPEC;

    snprintf (portstr, sizeof(portstr), "%d", ntohs(port));

    error = getaddrinfo (NULL, portstr, &hints, &ai);
    if (error)
	errx (1, "getaddrinfo: %s", gai_strerror (error));

    mini_inetd_addrinfo(ai, ret_socket);

    freeaddrinfo(ai);
}

