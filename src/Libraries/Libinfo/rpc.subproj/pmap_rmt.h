/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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
 *	from: @(#)pmap_rmt.h 1.2 88/02/08 SMI 
 *	from: @(#)pmap_rmt.h	2.1 88/07/29 4.0 RPCSRC
 *	$Id: pmap_rmt.h,v 1.3 2004/10/28 21:58:23 emoy Exp $
 */

/*
 * Structures and XDR routines for parameters to and replies from
 * the portmapper remote-call-service.
 *
 * Copyright (C) 1986, Sun Microsystems, Inc.
 */

#ifndef _RPC_PMAPRMT_H
#define _RPC_PMAPRMT_H
#include <sys/cdefs.h>
#include <rpc/xdr.h>

struct rmtcallargs {
#ifdef __LP64__
	unsigned int prog, vers, proc, arglen;
#else
	unsigned long prog, vers, proc, arglen;
#endif
	caddr_t args_ptr;
	xdrproc_t xdr_args;
};

struct rmtcallres {
#ifdef __LP64__
	unsigned int *port_ptr;
	unsigned int resultslen;
#else
	unsigned long *port_ptr;
	unsigned long resultslen;
#endif
	caddr_t results_ptr;
	xdrproc_t xdr_results;
};

__BEGIN_DECLS
extern bool_t xdr_rmtcall_args	__P((XDR *, struct rmtcallargs *));
extern bool_t xdr_rmtcallres	__P((XDR *, struct rmtcallres *));
__END_DECLS

#endif /* !_RPC_PMAPRMT_H */
