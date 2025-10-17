/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
 * Sun RPC is a product of Sun Microsystems, Inc. and is provided for
 * unrestricted use provided that this legend is included on all tape
 * media and as a part of the software program in whole or part.  Users
 * may copy or modify Sun RPC without charge, but are not authorized
 * to license or distribute it to anyone else except as part of a product or
 * program developed by the user.
 * 
 * SUN RPC IS PROVIDED AS IS WITH NO WARRANTIES OF ANY KIND INCLUDING THE
 * WARRANTIES OF DESIGN, MERCHANTIBILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE, OR ARISING FROM A COURSE OF DEALING, USAGE OR TRADE PRACTICE.
 * 
 * Sun RPC is provided with no support and without any obligation on the
 * part of Sun Microsystems, Inc. to assist in its use, correction,
 * modification or enhancement.
 * 
 * SUN MICROSYSTEMS, INC. SHALL HAVE NO LIABILITY WITH RESPECT TO THE
 * INFRINGEMENT OF COPYRIGHTS, TRADE SECRETS OR ANY PATENTS BY SUN RPC
 * OR ANY PART THEREOF.
 * 
 * In no event will Sun Microsystems, Inc. be liable for any lost revenue
 * or profits or other special, indirect and consequential damages, even if
 * Sun has been advised of the possibility of such damages.
 * 
 * Sun Microsystems, Inc.
 * 2550 Garcia Avenue
 * Mountain View, California  94043
 *
 *	from: @(#)pmap_prot.h 1.14 88/02/08 SMI 
 *	from: @(#)pmap_prot.h	2.1 88/07/29 4.0 RPCSRC
 *	$Id: pmap_prot.h,v 1.3 2004/10/28 21:58:22 emoy Exp $
 */

/*
 * pmap_prot.h
 * Protocol for the local binder service, or pmap.
 *
 * Copyright (C) 1984, Sun Microsystems, Inc.
 *
 * The following procedures are supported by the protocol:
 *
 * PMAPPROC_NULL() returns ()
 * 	takes nothing, returns nothing
 *
 * PMAPPROC_SET(struct pmap) returns (bool_t)
 * 	TRUE is success, FALSE is failure.  Registers the tuple
 *	[prog, vers, prot, port].
 *
 * PMAPPROC_UNSET(struct pmap) returns (bool_t)
 *	TRUE is success, FALSE is failure.  Un-registers pair
 *	[prog, vers].  prot and port are ignored.
 *
 * PMAPPROC_GETPORT(struct pmap) returns (long unsigned).
 *	0 is failure.  Otherwise returns the port number where the pair
 *	[prog, vers] is registered.  It may lie!
 *
 * PMAPPROC_DUMP() RETURNS (struct pmaplist *)
 *
 * PMAPPROC_CALLIT(unsigned, unsigned, unsigned, string<>)
 * 	RETURNS (port, string<>);
 * usage: encapsulatedresults = PMAPPROC_CALLIT(prog, vers, proc, encapsulatedargs);
 * 	Calls the procedure on the local machine.  If it is not registered,
 *	this procedure is quite; ie it does not return error information!!!
 *	This procedure only is supported on rpc/udp and calls via
 *	rpc/udp.  This routine only passes null authentication parameters.
 *	This file has no interface to xdr routines for PMAPPROC_CALLIT.
 *
 * The service supports remote procedure calls on udp/ip or tcp/ip socket 111.
 */

#ifndef _RPC_PMAPPROT_H
#define _RPC_PMAPPROT_H
#include <sys/cdefs.h>
#include <rpc/xdr.h>

#define PMAPPORT		((unsigned short)111)
#ifdef __LP64__
#define PMAPPROG		((unsigned int)100000)
#define PMAPVERS		((unsigned int)2)
#define PMAPVERS_PROTO		((unsigned int)2)
#define PMAPVERS_ORIG		((unsigned int)1)
#define PMAPPROC_NULL		((unsigned int)0)
#define PMAPPROC_SET		((unsigned int)1)
#define PMAPPROC_UNSET		((unsigned int)2)
#define PMAPPROC_GETPORT	((unsigned int)3)
#define PMAPPROC_DUMP		((unsigned int)4)
#define PMAPPROC_CALLIT		((unsigned int)5)
#else
#define PMAPPROG		((unsigned long)100000)
#define PMAPVERS		((unsigned long)2)
#define PMAPVERS_PROTO		((unsigned long)2)
#define PMAPVERS_ORIG		((unsigned long)1)
#define PMAPPROC_NULL		((unsigned long)0)
#define PMAPPROC_SET		((unsigned long)1)
#define PMAPPROC_UNSET		((unsigned long)2)
#define PMAPPROC_GETPORT	((unsigned long)3)
#define PMAPPROC_DUMP		((unsigned long)4)
#define PMAPPROC_CALLIT		((unsigned long)5)
#endif

struct pmap {
#ifdef __LP64__
	unsigned int pm_prog;
	unsigned int pm_vers;
	unsigned int pm_prot;
	unsigned int pm_port;
#else
	long unsigned pm_prog;
	long unsigned pm_vers;
	long unsigned pm_prot;
	long unsigned pm_port;
#endif
};

struct pmaplist {
	struct pmap	pml_map;
	struct pmaplist *pml_next;
};

__BEGIN_DECLS
extern bool_t xdr_pmap		__P((XDR *, struct pmap *));
extern bool_t xdr_pmaplist	__P((XDR *, struct pmaplist **));
__END_DECLS

#endif /* !_RPC_PMAPPROT_H */
