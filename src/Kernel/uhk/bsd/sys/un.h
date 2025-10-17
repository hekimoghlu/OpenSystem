/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
/*
 * Copyright (c) 1982, 1986, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *	@(#)un.h	8.3 (Berkeley) 2/19/95
 */

#ifndef _SYS_UN_H_
#define _SYS_UN_H_

#include <sys/appleapiopts.h>
#include <sys/cdefs.h>
#include <sys/_types.h>

/* [XSI] The sa_family_t type shall be defined as described in <sys/socket.h> */
#include <sys/_types/_sa_family_t.h>

/*
 * [XSI] Definitions for UNIX IPC domain.
 */
struct  sockaddr_un {
	unsigned char   sun_len;        /* sockaddr len including null */
	sa_family_t     sun_family;     /* [XSI] AF_UNIX */
	char            sun_path[104];  /* [XSI] path name (gag) */
};

#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)

/* Level number of get/setsockopt for local domain sockets */
#define SOL_LOCAL               0

/* Socket options. */
#define LOCAL_PEERCRED          0x001           /* retrieve peer credentials */
#define LOCAL_PEERPID           0x002           /* retrieve peer pid */
#define LOCAL_PEEREPID          0x003           /* retrieve eff. peer pid */
#define LOCAL_PEERUUID          0x004           /* retrieve peer UUID */
#define LOCAL_PEEREUUID         0x005           /* retrieve eff. peer UUID */
#define LOCAL_PEERTOKEN         0x006           /* retrieve peer audit token */

#endif  /* (!_POSIX_C_SOURCE || _DARWIN_C_SOURCE) */


#ifdef KERNEL
#ifdef PRIVATE
#include <kern/locks.h>
__BEGIN_DECLS
struct mbuf;
struct socket;
struct sockopt;

int     uipc_usrreq(struct socket *so, int req, struct mbuf *m,
    struct mbuf *nam, struct mbuf *control);
int     uipc_ctloutput(struct socket *so, struct sockopt *sopt);
int     unp_connect2(struct socket *so, struct socket *so2);
void    unp_dispose(struct mbuf *m);
int     unp_externalize(struct mbuf *rights);
void    unp_init(void);
extern  struct pr_usrreqs uipc_usrreqs;
int     unp_lock(struct socket *, int, void *);
int     unp_unlock(struct socket *, int, void *);
lck_mtx_t* unp_getlock(struct socket *, int);

#define UNP_FORGE_PATH(sun, len) ({                                     \
	__unsafe_forge_bidi_indexable(char *, &sun->sun_path, len);     \
})

__END_DECLS
#endif /* PRIVATE */
#else /* !KERNEL */

#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)
/* actual length of an initialized sockaddr_un */
#define SUN_LEN(su) \
	(sizeof(*(su)) - sizeof((su)->sun_path) + strlen((su)->sun_path))
#endif  /* (!_POSIX_C_SOURCE || _DARWIN_C_SOURCE) */
#endif /* KERNEL */

#endif /* !_SYS_UN_H_ */
