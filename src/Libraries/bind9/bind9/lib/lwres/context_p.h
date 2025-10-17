/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
/* $Id: context_p.h,v 1.19 2008/12/17 23:47:58 tbox Exp $ */

#ifndef LWRES_CONTEXT_P_H
#define LWRES_CONTEXT_P_H 1

/*! \file */

/*@{*/
/**
 * Helper functions, assuming the context is always called "ctx" in
 * the scope these functions are called from.
 */
#define CTXMALLOC(len)		ctx->malloc(ctx->arg, (len))
#define CTXFREE(addr, len)	ctx->free(ctx->arg, (addr), (len))
/*@}*/

#define LWRES_DEFAULT_TIMEOUT	120	/* 120 seconds for a reply */

/**
 * Not all the attributes here are actually settable by the application at
 * this time.
 */
struct lwres_context {
	unsigned int		timeout;	/*%< time to wait for reply */
	lwres_uint32_t		serial;		/*%< serial number state */

	/*
	 * For network I/O.
	 */
	int			sock;		/*%< socket to send on */
	lwres_addr_t		address;	/*%< address to send to */
	int			use_ipv4;	/*%< use IPv4 transaction */
	int			use_ipv6;	/*%< use IPv6 transaction */

	/*@{*/
	/*
	 * Function pointers for allocating memory.
	 */
	lwres_malloc_t		malloc;
	lwres_free_t		free;
	void		       *arg;
	/*@}*/

	/*%
	 * resolv.conf-like data
	 */
	lwres_conf_t		confdata;
};

#endif /* LWRES_CONTEXT_P_H */
