/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
 * Copyright (c) 1992, 1993 Theo de Raadt <deraadt@theos.com>
 * All rights reserved.
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
 *	This product includes software developed by Theo de Raadt.
 * 4. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#if defined(LIBC_SCCS) && !defined(lint)
static char *rcsid = "$OpenBSD: yp_master.c,v 1.5 1996/12/03 08:20:05 deraadt Exp $";
#endif /* LIBC_SCCS and not lint */

#include "libinfo_common.h"

#include <sys/param.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <sys/uio.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <rpc/rpc.h>
#include <rpc/xdr.h>
#include <rpcsvc/yp.h>
#include <rpcsvc/ypclnt.h>
#include "ypinternal.h"

LIBINFO_EXPORT
int
yp_master(indomain, inmap, outname)
	const char     *indomain;
	const char     *inmap;
	char          **outname;
{
	struct dom_binding *ysd;
	struct ypresp_master yprm;
	struct ypreq_nokey yprnk;
	struct timeval  tv;
	int tries = 0, r;
	static int proto = YP_BIND_UDP;

	if (indomain == NULL || *indomain == '\0' ||
	    strlen(indomain) > YPMAXDOMAIN || inmap == NULL ||
	    *inmap == '\0' || strlen(inmap) > YPMAXMAP || outname == NULL)
		return YPERR_BADARGS;

again:
	if (_yp_dobind(indomain, &ysd) != 0)
		return YPERR_DOMAIN;

	tv.tv_sec = _yplib_timeout;
	tv.tv_usec = 0;

	yprnk.domain = (char *)indomain;
	yprnk.map = (char *)inmap;

	(void)memset(&yprm, 0, sizeof yprm);

	r = clnt_call(ysd->dom_client, YPPROC_MASTER, (xdrproc_t)xdr_ypreq_nokey, &yprnk, (xdrproc_t)xdr_ypresp_master, &yprm, tv);
	if (r != RPC_SUCCESS)
	{
		if (tries++) clnt_perror(ysd->dom_client, "yp_master: clnt_call");

		if (proto == YP_BIND_UDP) proto = YP_BIND_TCP;
		else proto = YP_BIND_UDP;
		ysd->dom_vers = proto;

		goto again;
	}

	if (!(r = ypprot_err(yprm.stat))) {
		if ((*outname = strdup(yprm.peer)) == NULL)
			r = YPERR_RESRC;
	}
	xdr_free((xdrproc_t)xdr_ypresp_master, (char *) &yprm);
	_yp_unbind(ysd);
	return r;
}
