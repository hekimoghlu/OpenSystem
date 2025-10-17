/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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
 * Support for socket filter kernel extensions
 */

#ifndef NET_KEXT_NET_H
#define NET_KEXT_NET_H

#include <sys/appleapiopts.h>

#include <sys/queue.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#ifdef BSD_KERNEL_PRIVATE
/*
 * Internal implementation bits
 */
#include <sys/kpi_socketfilter.h>

struct socket;
struct sockopt;
struct inpcb;

/* Private, internal implementation functions */
extern int      sflt_permission_check(struct inpcb *inp);
extern void     sflt_initsock(struct socket *so);
extern void     sflt_termsock(struct socket *so);
extern errno_t  sflt_attach_internal(struct socket *so, sflt_handle     handle);
extern void     sflt_notify(struct socket *so, sflt_event_t event, void *param);
extern int      sflt_ioctl(struct socket *so, u_long cmd, caddr_t __sized_by(IOCPARM_LEN(cmd)) data);
extern int      sflt_bind(struct socket *so, const struct sockaddr *nam);
extern int      sflt_listen(struct socket *so);
extern int      sflt_accept(struct socket *head, struct socket *so,
    const struct sockaddr *local,
    const struct sockaddr *remote);
extern int      sflt_getsockname(struct socket *so, struct sockaddr **local);
extern int      sflt_getpeername(struct socket *so, struct sockaddr **remote);
extern int      sflt_connectin(struct socket *head,
    const struct sockaddr *remote);
extern int      sflt_connectout(struct socket *so, const struct sockaddr *nam);
extern int      sflt_setsockopt(struct socket *so, struct sockopt *sopt);
extern int      sflt_getsockopt(struct socket *so, struct sockopt *sopt);
extern int      sflt_data_out(struct socket *so, const struct sockaddr  *to,
    mbuf_t *data, mbuf_t *control, sflt_data_flag_t flags);
extern int      sflt_data_in(struct socket *so, const struct sockaddr *from,
    mbuf_t *data, mbuf_t *control, sflt_data_flag_t flags);

#endif /* BSD_KERNEL_PRIVATE */

#define NFF_BEFORE              0x01
#define NFF_AFTER               0x02

#define NKE_OK                  0
#define NKE_REMOVE              (-1)

/*
 * Interface structure for inserting an installed socket NKE into an
 *  existing socket.
 * 'handle' is the NKE to be inserted, 'where' is an insertion point,
 *  and flags dictate the position of the to-be-inserted NKE relative to
 *  the 'where' NKE.  If the latter is NULL, the flags indicate "first"
 *  or "last"
 */
#pragma pack(4)

struct so_nke {
	unsigned int nke_handle;
	unsigned int nke_where;
	int nke_flags; /* NFF_BEFORE, NFF_AFTER: net/kext_net.h */
	u_int32_t reserved[4];  /* for future use */
};

#pragma pack()
#endif /* NET_KEXT_NET_H */
