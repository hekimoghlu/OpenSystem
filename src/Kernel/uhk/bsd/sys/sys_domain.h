/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#ifndef _SYSTEM_DOMAIN_H_
#define _SYSTEM_DOMAIN_H_

#include <sys/appleapiopts.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#ifdef KERNEL_PRIVATE
#include <sys/sysctl.h>
#endif /* KERNEL_PRIVATE */

/* Kernel Events Protocol */
#define SYSPROTO_EVENT          1       /* kernel events protocol */

/* Kernel Control Protocol */
#define SYSPROTO_CONTROL        2       /* kernel control protocol */
#define AF_SYS_CONTROL          2       /* corresponding sub address type */

/* System family socket address */
struct sockaddr_sys {
	u_char          ss_len;         /* sizeof(struct sockaddr_sys) */
	u_char          ss_family;      /* AF_SYSTEM */
	u_int16_t       ss_sysaddr;     /* protocol address in AF_SYSTEM */
	u_int32_t       ss_reserved[7]; /* reserved to the protocol use */
};

#ifdef PRIVATE
struct  xsystmgen {
	u_int32_t       xg_len; /* length of this structure */
	u_int64_t       xg_count;       /* number of PCBs at this time */
	u_int64_t       xg_gen; /* generation count at this time */
	u_int64_t       xg_sogen;       /* current socket generation count */
};
#endif /* PRIVATE */

#ifdef KERNEL_PRIVATE

extern struct domain *systemdomain;

SYSCTL_DECL(_net_systm);

/* built in system domain protocols init function */
__BEGIN_DECLS
void kern_event_init(struct domain *);
void kern_control_init(struct domain *);
__END_DECLS
#endif /* KERNEL_PRIVATE */

#endif /* _SYSTEM_DOMAIN_H_ */
