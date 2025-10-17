/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
 * Mach Operating System
 * Copyright (c) 1987 Carnegie-Mellon University
 * All rights reserved.  The CMU software License Agreement specifies
 * the terms and conditions for use and redistribution.
 */

/*
 * Random definitions for the network service that everyone needs!
 */

/*
 * HISTORY:
 * 27-Mar-90  Gregg Kellogg (gk) at NeXT
 *	include <sys/netport.h> rather than <sys/ipc_netport.h>
 *
 * 24-Aug-88  Daniel Julin (dpj) at Carnegie-Mellon University
 *	Replace sys/mach_ipc_netport.h with kern/ipc_netport.h. Sigh.
 *
 * 24-May-88  Daniel Julin (dpj) at Carnegie-Mellon University
 *	Replace mach_ipc_vmtp.h with mach_ipc_netport.h.
 *
 *  4-Sep-87  Daniel Julin (dpj) at Carnegie-Mellon University
 *	Fixed for new kernel include files which declare a lot
 *	of network server stuff internally, because of the NETPORT
 *	option.
 *
 *  5-Nov-86  Robert Sansom (rds) at Carnegie-Mellon University
 *	Started.
 *
 */

#ifndef _NM_DEFS_
#define _NM_DEFS_

/*
 * netaddr_t is declared with the kernel files,
 * in <sys/netport.h>.
 */
#include        <sys/netport.h>

#ifdef  notdef
typedef unsigned long   netaddr_t;
#endif  /* notdef */

typedef union {
	struct {
		unsigned char ia_net_owner;
		unsigned char ia_net_node_type;
		unsigned char ia_host_high;
		unsigned char ia_host_low;
	} ia_bytes;
	netaddr_t ia_netaddr;
} ip_addr_t;

#endif  /* _NM_DEFS_ */
