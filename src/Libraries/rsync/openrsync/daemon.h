/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#ifndef DAEMON_H
#define	DAEMON_H

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include <getopt.h>
#include <netdb.h>
#include <stdbool.h>

#include "extern.h"

#define	RSYNCD_MUNGE_PREFIX	"/rsyncd-munged/"

/*
 * Memory legend:
 *
 * (c) Allocated within config, or an option pointer -- do not free
 * (f) Allocated independently, child should free
 */

struct daemon_refused {
	const struct option	* const *refused_lopts;	/* (f) */
	char			 *refused_shopts;	/* (f) */
	size_t			 refused_loptsz;
};

enum daemon_state : unsigned char {
	DSTATE_INIT,
	DSTATE_CLIENT_CONTROL,
	DSTATE_RUNNING,
};

struct daemon_role {
	struct role		 role;
	char			 client_host[NI_MAXHOST]; /* hostname */
	char			 client_addr[INET6_ADDRSTRLEN]; /* addr */
	struct sockaddr		*client_sa;
	char			*auth_user;	/* (f) auth user */
	const char		*cfg_file;	/* (c) daemon config file */
	char			*motd_file;	/* (f) client motd */
	const char		*module;	/* (c) module */
	const char		*module_path;	/* (c) module path */
	struct daemon_cfg	*dcfg;		/* (f) daemon config */
	const char		*pid_file;	/* (c) daemon pidfile path */
	FILE			*pidfp;		/* (f) daemon pidfile */
	int			 prexfer_pipe;	/* (f) pre-xfer pipe */
	pid_t			 prexfer_pid;	/* pre-xfer exec process */
	int			 lockfd;
	id_t			 uid;		/* setuid if root */
	id_t			 gid;		/* setgid if root */
	int			 client;
	enum daemon_state	 dstate;
	bool			 do_setid;	/* do setuid/setgid */
	struct daemon_refused	 refused;
	bool			 using_logfile;	/* "log file" specified */
	bool			 socket_initiator;	/* socket-initiated */
};

int	daemon_apply_chmod(struct sess *, const char *, struct opts *);
int	daemon_apply_chrootopts(struct sess *, const char *, struct opts *,
	    int);
int	daemon_apply_ignoreopts(struct sess *, const char *, struct opts *);
int	daemon_apply_xferlog(struct sess *, const char *, struct opts *);
int	daemon_chuser_setup(struct sess *, const char *);
int	daemon_chuser(struct sess *, const char *);
void	daemon_client_error(struct sess *, const char *, ...);
int	daemon_configure_filters(struct sess *, const char *);
int	daemon_connection_allowed(struct sess *, const char *);
int	daemon_connection_limited(struct sess *, const char *);
int	daemon_do_execcmds(struct sess *, const char *);
int	daemon_finish_handshake(struct sess *);
int	daemon_finish_prexfer(struct sess *, const char *, const char *,
	    size_t);
int	daemon_fill_hostinfo(struct sess *, const char *,
	    const struct sockaddr *, socklen_t);
int	daemon_install_symlink_filter(struct sess *, const char *, int);
int	daemon_limit_verbosity(struct sess *, const char *);
void	daemon_normalize_path(const char *, size_t, char *);
void	daemon_normalize_paths(const char *, int, char *[]);
int	daemon_open_logfile(struct sess *, const char *, bool);
int	daemon_operation_allowed(struct sess *, const struct opts *,
	    const char *, int);
int	daemon_parse_refuse(struct sess *, const char *);
int	daemon_setup_logfile(struct sess *, const char *);

#endif /* !DAEMON_H */
