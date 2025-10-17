/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#ifndef _EXTERN_H_
#define _EXTERN_H_

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif

#include <stdio.h>
#include <stdarg.h>
#ifdef HAVE_PWD_H
#include <pwd.h>
#endif

#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif

#ifndef NBBY
#define NBBY CHAR_BIT
#endif

void	abor(void);
void	blkfree(char **);
char  **copyblk(char **);
void	cwd(const char *);
void	do_delete(char *);
void	dologout(int);
void	eprt(char *);
void	epsv(char *);
void	fatal(char *);
int	filename_check(char *);
int	ftpd_pclose(FILE *);
FILE   *ftpd_popen(char *, char *, int, int);
char   *ftpd_getline(char *, int);
void	ftpd_logwtmp(char *, char *, char *);
void	lreply(int, const char *, ...)
    __attribute__ ((format (printf, 2, 3)));
void	makedir(char *);
void	nack(char *);
void	nreply(const char *, ...)
    __attribute__ ((format (printf, 1, 2)));
void	pass(char *);
void	pasv(void);
void	perror_reply(int, const char *);
void	pwd(void);
void	removedir(char *);
void	renamecmd(char *, char *);
char   *renamefrom(char *);
void	reply(int, const char *, ...)
    __attribute__ ((format (printf, 2, 3)));
void	retrieve(const char *, char *);
void	send_file_list(char *);
void	setproctitle(const char *, ...)
    __attribute__ ((format (printf, 1, 2)));
void	statcmd(void);
void	statfilecmd(char *);
void	do_store(char *, char *, int);
void	upper(char *);
void	user(char *);
void	yyerror(char *);

void	list_file(char*);

void	kauth(char *, char*);
void	klist(void);
void	cond_kdestroy(void);
void	kdestroy(void);
void	krbtkfile(const char *tkfile);
void	afslog(const char *, int);
void	afsunlog(void);

extern int do_destroy_tickets;
extern char *k5ccname;

int	find(char *);

int	builtin_ls(FILE*, const char*);

int	do_login(int code, char *passwd);
int	klogin(char *name, char *password);

const char *ftp_rooted(const char *path);

extern struct sockaddr *ctrl_addr, *his_addr;
extern char hostname[];

extern	struct sockaddr *data_dest;
extern	int logged_in;
extern	struct passwd *pw;
extern	int guest;
extern  int dochroot;
extern	int logging;
extern	int type;
extern off_t file_size;
extern off_t byte_count;
extern	int ccc_passed;

extern	int form;
extern	int debug;
extern	int ftpd_timeout;
extern	int maxtimeout;
extern  int pdata;
extern	char hostname[], remotehost[];
extern	char proctitle[];
extern	int usedefault;
extern  char tmpline[];
extern  int paranoid;

#endif /* _EXTERN_H_ */
