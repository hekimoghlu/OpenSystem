/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
/* $Id$ */

#include <setjmp.h>
#include <stdlib.h>
#ifdef TIME_WITH_SYS_TIME
#include <sys/time.h>
#include <time.h>
#elif defined(HAVE_SYS_TIME_H)
#include <sys/time.h>
#else
#include <time.h>
#endif
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif

void    abort_remote (FILE *);
void    abortpt (int);
void    abortrecv (int);
void	account (int, char **);
int	another (int *, char ***, char *);
void	blkfree (char **);
void	cd (int, char **);
void	cdup (int, char **);
void	changetype (int, int);
void	cmdabort (int);
void	cmdscanner (int);
int	command (char *fmt, ...)
    __attribute__ ((format (printf, 1,2)));
int	confirm (char *, char *);
FILE   *dataconn (const char *);
void	delete (int, char **);
void	disconnect (int, char **);
void	do_chmod (int, char **);
void	do_umask (int, char **);
void	domacro (int, char **);
char   *domap (char *);
void	doproxy (int, char **);
char   *dotrans (char *);
int     empty (fd_set *, int);
void	fatal (char *);
void	get (int, char **);
struct cmd *getcmd (char *);
int	getit (int, char **, int, char *);
int	getreply (int);
int	globulize (char **);
char   *gunique (char *);
void	help (int, char **);
char   *hookup (const char *, int);
void	ftp_idle (int, char **);
int     initconn (void);
void	intr (int);
void	lcd (int, char **);
int	login (char *);
RETSIGTYPE	lostpeer (int);
void	ls (int, char **);
void	macdef (int, char **);
void	makeargv (void);
void	makedir (int, char **);
void	mdelete (int, char **);
void	mget (int, char **);
void	mls (int, char **);
void	modtime (int, char **);
void	mput (int, char **);
char   *onoff (int);
void	newer (int, char **);
void    proxtrans (char *, char *, char *);
void    psabort (int);
void    pswitch (int);
void    ptransfer (char *, long, struct timeval *, struct timeval *);
void	put (int, char **);
void	pwd (int, char **);
void	quit (int, char **);
void	quote (int, char **);
void	quote1 (char *, int, char **);
void    recvrequest (char *, char *, char *, char *, int, int);
void	reget (int, char **);
char   *remglob (char **, int);
void	removedir (int, char **);
void	renamefile (int, char **);
void    reset (int, char **);
void	restart (int, char **);
void	rmthelp (int, char **);
void	rmtstatus (int, char **);
int	ruserpassword (char *, char **, char **, char **);
void    sendrequest (char *, char *, char *, char *, int);
void	setascii (int, char **);
void	setbell (int, char **);
void	setbinary (int, char **);
void	setcase (int, char **);
void	setcr (int, char **);
void	setdebug (int, char **);
void	setform (int, char **);
void	setftmode (int, char **);
void	setglob (int, char **);
void	sethash (int, char **);
void	setnmap (int, char **);
void	setntrans (int, char **);
void	setpassive (int, char **);
void	setpeer (int, char **);
void	setport (int, char **);
void	setprompt (int, char **);
void	setrunique (int, char **);
void	setstruct (int, char **);
void	setsunique (int, char **);
void	settenex (int, char **);
void	settrace (int, char **);
void	settype (int, char **);
void	setverbose (int, char **);
void	shell (int, char **);
void	site (int, char **);
void	sizecmd (int, char **);
char   *slurpstring (void);
void	status (int, char **);
void	syst (int, char **);
void    tvsub (struct timeval *, struct timeval *, struct timeval *);
void	user (int, char **);

extern jmp_buf	abortprox;
extern int	abrtflag;
extern struct	cmd cmdtab[];
extern FILE	*cout;
extern int	data;
extern char    *home;
extern jmp_buf	jabort;
extern int	proxy;
extern char	reply_string[];
extern off_t	restart_point;
extern int	NCMDS;

extern char 	username[32];
extern char	myhostname[];
extern char	*mydomain;

void afslog (int, char **);
void kauth (int, char **);
void kdestroy (int, char **);
void klist (int, char **);
void krbtkfile (int, char **);
