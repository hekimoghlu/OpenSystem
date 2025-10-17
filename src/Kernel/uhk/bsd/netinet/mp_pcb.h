/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#ifndef _NETINET_MP_PCB_H_
#define _NETINET_MP_PCB_H_

#ifdef BSD_KERNEL_PRIVATE
#include <netinet/in_pcb.h>

#include <sys/domain.h>
#include <sys/protosw.h>
#include <sys/socketvar.h>
#include <sys/types.h>
#include <sys/queue.h>
#include <kern/locks.h>

/* Keep in sync with bsd/dev/dtrace/scripts/mptcp.d */
typedef enum mppcb_state {
	MPPCB_STATE_INUSE       = 1,
	MPPCB_STATE_DEAD        = 2,
} mppcb_state_t;

/*
 * Multipath Protocol Control Block
 */
struct mppcb {
	TAILQ_ENTRY(mppcb)      mpp_entry;      /* glue to all PCBs */
	decl_lck_mtx_data(, mpp_lock);          /* per PCB lock */
	struct mppcbinfo        *mpp_pcbinfo;   /* PCB info */
	struct mptses           *mpp_pcbe;      /* ptr to MPTCP-session */
	struct socket           *mpp_socket;    /* back pointer to socket */
	uint32_t                mpp_flags;      /* PCB flags */
	mppcb_state_t           mpp_state;      /* PCB state */
	int32_t                 mpp_inside;     /* Indicates whether or not a thread is processing MPTCP */

#if NECP
	uuid_t necp_client_uuid;
	struct inp_necp_attributes inp_necp_attributes;
	void (*necp_cb)(void *, int, uint32_t, uint32_t, bool *);
#endif
};

static inline struct mppcb *
mpsotomppcb(struct socket *mp_so)
{
	VERIFY(SOCK_DOM(mp_so) == PF_MULTIPATH);
	return (struct mppcb *)mp_so->so_pcb;
}

/* valid values for mpp_flags */
#define MPP_ATTACHED            0x001
#define MPP_INSIDE_OUTPUT       0x002           /* MPTCP-stack is inside mptcp_subflow_output */
#define MPP_INSIDE_INPUT        0x004           /* MPTCP-stack is inside mptcp_subflow_input */
#define MPP_INPUT_HANDLE        0x008           /* MPTCP-stack is handling input */
#define MPP_WUPCALL             0x010           /* MPTCP-stack is handling a read upcall */
#define MPP_SHOULD_WORKLOOP     0x020           /* MPTCP-stack should call the workloop function */
#define MPP_SHOULD_RWAKEUP      0x040           /* MPTCP-stack should call sorwakeup */
#define MPP_SHOULD_WWAKEUP      0x080           /* MPTCP-stack should call sowwakeup */
#define MPP_CREATE_SUBFLOWS     0x100           /* This connection needs to create subflows */
#define MPP_INSIDE_SETGETOPT    0x200           /* MPTCP-stack is inside mptcp_setopt/mptcp_getopt */

static inline boolean_t
mptcp_should_defer_upcall(struct mppcb *mpp)
{
	return !!(mpp->mpp_flags & (MPP_INSIDE_OUTPUT | MPP_INSIDE_INPUT | MPP_INPUT_HANDLE | MPP_WUPCALL));
}

/*
 * Multipath PCB Information
 */
struct mppcbinfo {
	TAILQ_ENTRY(mppcbinfo)  mppi_entry;     /* glue to all PCB info */
	TAILQ_HEAD(, mppcb)     mppi_pcbs;      /* list of PCBs */
	uint32_t                mppi_count;     /* # of PCBs in list */
	lck_attr_t              mppi_lock_attr; /* lock attr */
	struct mppcb         *(*mppi_alloc)(void);
	void                  (*mppi_free)(struct mppcb *);
	lck_grp_t              *mppi_lock_grp;  /* lock grp */
	decl_lck_mtx_data(, mppi_lock);         /* global PCB lock */
	uint32_t (*mppi_gc)(struct mppcbinfo *); /* garbage collector func */
	uint32_t (*mppi_timer)(struct mppcbinfo *); /* timer func */
};

__BEGIN_DECLS
extern void mp_pcbinfo_attach(struct mppcbinfo *);
extern int mp_pcbinfo_detach(struct mppcbinfo *);
extern int mp_pcballoc(struct socket *, struct mppcbinfo *);
extern void mp_pcbdetach(struct socket *);
extern void mptcp_pcbdispose(struct mppcb *);
extern void mp_gc_sched(void);
extern void mptcp_timer_sched(void);
extern void mptcp_handle_deferred_upcalls(struct mppcb *mpp, uint32_t flag);
extern int mp_getsockaddr(struct socket *mp_so, struct sockaddr **nam);
extern int mp_getpeeraddr(struct socket *mp_so, struct sockaddr **nam);
#if NECP
extern int necp_client_register_multipath_cb(pid_t pid, uuid_t client_id, struct mppcb *mpp);
extern void necp_mppcb_dispose(struct mppcb *mpp);
#endif
__END_DECLS

#endif /* BSD_KERNEL_PRIVATE */
#endif /* !_NETINET_MP_PCB_H_ */
