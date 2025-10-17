/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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
 * Copyright (c) 1998 by Apple Computer, Inc.
 * Portions Copyright (c) 1988 by Sun Microsystems, Inc.
 * Portions Copyright (c) 1988 The Regents of the University of California.
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
 */


/* update a user's password in NIS. This was based on the Sun implementation
 * we used in NEXTSTEP, although I've added some stuff from OpenBSD. And
 * it uses the API to support Rhapsody's proprietry infosystem switch.
 * lukeh
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pwd.h>
#include <netinet/in.h>
#include <rpc/rpc.h>
#include <rpc/xdr.h>
#include <rpc/pmap_clnt.h>
#include <rpcsvc/yp_prot.h>
#include <rpcsvc/ypclnt.h>
#include <rpcsvc/yppasswd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <errno.h>

#include "passwd.h"

static struct passwd *ypgetpwnam(char *name, char *domain);

int
nis_check_passwd(char *uname, char *domain)
{
	int port;
	char *master;
	struct passwd *pwd;

	if (domain == NULL)
	{
		if (yp_get_default_domain(&domain) != 0)
		{
			(void)fprintf(stderr, "can't get domain\n");
			exit(1);
		}
	}

	if (yp_master(domain, "passwd.byname", &master) != 0)
	{
		(void)fprintf(stderr, "can't get master for passwd file\n");
		exit(1);
	}

	port = getrpcport(master, YPPASSWDPROG, YPPASSWDPROC_UPDATE,
		IPPROTO_UDP);
	if (port == 0)
	{
		(void)fprintf(stderr, "%s is not running yppasswd daemon\n",
			      master);
		exit(1);
	}
	if (port >= IPPORT_RESERVED)
	{
		(void)fprintf(stderr,
		    "yppasswd daemon is not running on privileged port\n");
		exit(1);
	}

	pwd = ypgetpwnam(uname, domain);
	if (pwd == NULL)
	{
		(void)fprintf(stderr, "user %s not found\n", uname);
		exit(1);
	}

	checkpasswd(uname, pwd->pw_passwd);
	return(0);
}

static char *
pwskip(register char *p)
{
	while (*p && *p != ':' && *p != '\n')
		++p;
	if (*p)
		*p++ = 0;
	return (p);
}

static struct passwd *
interpret(struct passwd *pwent, char *line)
{
	register char	*p = line;

	pwent->pw_passwd = "*";
	pwent->pw_uid = 0;
	pwent->pw_gid = 0;
	pwent->pw_gecos = "";
	pwent->pw_dir = "";
	pwent->pw_shell = "";
#ifndef __SLICK__
	pwent->pw_change = 0;
	pwent->pw_expire = 0;
	pwent->pw_class = "";
#endif

	/* line without colon separators is no good, so ignore it */
	if(!strchr(p, ':'))
		return(NULL);

	pwent->pw_name = p;
	p = pwskip(p);
	pwent->pw_passwd = p;
	p = pwskip(p);
	pwent->pw_uid = (uid_t)strtoul(p, NULL, 10);
	p = pwskip(p);
	pwent->pw_gid = (gid_t)strtoul(p, NULL, 10);
	p = pwskip(p);
	pwent->pw_gecos = p;
	p = pwskip(p);
	pwent->pw_dir = p;
	p = pwskip(p);
	pwent->pw_shell = p;
	while (*p && *p != '\n')
		p++;
	*p = '\0';
	return (pwent);
}

static struct passwd *
ypgetpwnam(char *nam, char *domain)
{
	static struct passwd pwent;
	char *val;
	int reason, vallen;
	static char *__yplin = NULL;

	reason = yp_match(domain, "passwd.byname", nam, (int)strlen(nam),
	    &val, &vallen);
	switch(reason) {
	case 0:
		break;
	default:
		return (NULL);
		break;
	}
	val[vallen] = '\0';
	if (__yplin)
		free(__yplin);
	__yplin = (char *)malloc(vallen + 1);
	strcpy(__yplin, val);
	free(val);

	return(interpret(&pwent, __yplin));
}
