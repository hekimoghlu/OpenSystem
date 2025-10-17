/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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
#ifndef _SASLAUTHDMAIN_H
#define _SASLAUTHDMAIN_H

#include <sys/types.h>
#include "saslauthd.h"

/****************************************************************
 * Plug in some autoconf magic to determine what IPC method
 * to use.
 ****************************************************************/
#ifdef USE_DOORS
# define USE_DOORS_IPC
#else
# define USE_UNIX_IPC
#endif

/* AIX uses a slight variant of this */
#ifdef _AIX
# define SALEN_TYPE size_t
#else 
# define SALEN_TYPE int
#endif 

/* Define some macros. These help keep the ifdefs out of the
 * mainline code. */
#ifdef AUTH_SIA
#define SET_AUTH_PARAMETERS(argc, argv) set_auth_parameters(argc, argv)
#else
#define SET_AUTH_PARAMETERS(argc, argv)
#endif

/* file name defines - don't forget the '/' in these! */
#define PID_FILE		"/saslauthd.pid"    
#define PID_FILE_LOCK		"/saslauthd.pid.lock"
#define ACCEPT_LOCK_FILE	"/mux.accept"       
#define SOCKET_FILE		"/mux"              
#define DOOR_FILE		"/mux"              

/* login, pw, service, realm buffer size */
#define MAX_REQ_LEN		256     

/* socket backlog when supported */
#define SOCKET_BACKLOG  	32

/* saslauthd-main.c */
extern char	*do_auth(const char *, const char *,
			 const char *, const char *);
extern void	set_auth_mech(const char *);
extern void	set_max_procs(const char *);
extern void	set_mech_option(const char *);
extern void	set_run_path(const char *);
extern void	signal_setup();
extern void	detach_tty();
extern void	handle_sigchld();
extern void	server_exit();
extern pid_t	have_baby();

/* ipc api delcarations */
extern void	ipc_init();
extern void	ipc_loop();
extern void	ipc_cleanup();

#endif  /* _SASLAUTHDMAIN_H */
