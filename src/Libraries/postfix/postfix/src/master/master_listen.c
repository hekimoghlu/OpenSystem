/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
/* System library. */

#include <sys_defs.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <listen.h>
#include <mymalloc.h>
#include <stringops.h>
#include <inet_addr_list.h>
#include <set_eugid.h>
#include <set_ugid.h>
#include <iostuff.h>
#include <myaddrinfo.h>
#include <sock_addr.h>

/* Global library. */

#include <mail_params.h>

/* Application-specific. */

#include "master.h"

/* master_listen_init - enable connection requests */

void    master_listen_init(MASTER_SERV *serv)
{
    const char *myname = "master_listen_init";
    char   *end_point;
    int     n;
    MAI_HOSTADDR_STR hostaddr;
    struct sockaddr *sa;

    /*
     * Find out what transport we should use, then create one or more
     * listener sockets. Make the listener sockets non-blocking, so that
     * child processes don't block in accept() when multiple processes are
     * selecting on the same socket and only one of them gets the connection.
     */
    switch (serv->type) {

	/*
	 * UNIX-domain or stream listener endpoints always come as singlets.
	 */
    case MASTER_SERV_TYPE_UNIX:
	set_eugid(var_owner_uid, var_owner_gid);
	serv->listen_fd[0] =
	    LOCAL_LISTEN(serv->name, serv->max_proc > var_proc_limit ?
			 serv->max_proc : var_proc_limit, NON_BLOCKING);
	close_on_exec(serv->listen_fd[0], CLOSE_ON_EXEC);
	set_ugid(getuid(), getgid());
	break;

	/*
	 * FIFO listener endpoints always come as singlets.
	 */
    case MASTER_SERV_TYPE_FIFO:
	set_eugid(var_owner_uid, var_owner_gid);
	serv->listen_fd[0] = fifo_listen(serv->name, 0622, NON_BLOCKING);
	close_on_exec(serv->listen_fd[0], CLOSE_ON_EXEC);
	set_ugid(getuid(), getgid());
	break;

	/*
	 * INET-domain listener endpoints can be wildcarded (the default) or
	 * bound to specific interface addresses.
	 * 
	 * With dual-stack IPv4/6 systems it does not matter, we have to specify
	 * the addresses anyway, either explicit or wild-card.
	 */
    case MASTER_SERV_TYPE_INET:
	for (n = 0; n < serv->listen_fd_count; n++) {
	    sa = SOCK_ADDR_PTR(MASTER_INET_ADDRLIST(serv)->addrs + n);
	    SOCKADDR_TO_HOSTADDR(sa, SOCK_ADDR_LEN(sa), &hostaddr,
				 (MAI_SERVPORT_STR *) 0, 0);
	    end_point = concatenate(hostaddr.buf,
				    ":", MASTER_INET_PORT(serv), (char *) 0);
	    serv->listen_fd[n]
		= inet_listen(end_point, serv->max_proc > var_proc_limit ?
			      serv->max_proc : var_proc_limit, NON_BLOCKING);
	    close_on_exec(serv->listen_fd[n], CLOSE_ON_EXEC);
	    myfree(end_point);
	}
	break;

	/*
	 * Descriptor passing endpoints always come as singlets.
	 */
#ifdef MASTER_SERV_TYPE_PASS
    case MASTER_SERV_TYPE_PASS:
	set_eugid(var_owner_uid, var_owner_gid);
	serv->listen_fd[0] =
	    LOCAL_LISTEN(serv->name, serv->max_proc > var_proc_limit ?
			serv->max_proc : var_proc_limit, NON_BLOCKING);
	close_on_exec(serv->listen_fd[0], CLOSE_ON_EXEC);
	set_ugid(getuid(), getgid());
	break;
#endif
    default:
	msg_panic("%s: unknown service type: %d", myname, serv->type);
    }
}

/* master_listen_cleanup - disable connection requests */

void    master_listen_cleanup(MASTER_SERV *serv)
{
    const char *myname = "master_listen_cleanup";
    int     n;

    /*
     * XXX The listen socket is shared with child processes. Closing the
     * socket in the master process does not really disable listeners in
     * child processes. There seems to be no documented way to turn off a
     * listener. The 4.4BSD shutdown(2) man page promises an ENOTCONN error
     * when shutdown(2) is applied to a socket that is not connected.
     */
    for (n = 0; n < serv->listen_fd_count; n++) {
	if (close(serv->listen_fd[n]) < 0)
	    msg_warn("%s: close listener socket %d: %m",
		     myname, serv->listen_fd[n]);
	serv->listen_fd[n] = -1;
    }
}
