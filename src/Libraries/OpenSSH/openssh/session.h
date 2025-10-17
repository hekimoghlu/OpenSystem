/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
 * Copyright (c) 2000, 2001 Markus Friedl.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef SESSION_H
#define SESSION_H

#define TTYSZ 64
typedef struct Session Session;
struct Session {
	int	used;
	int	self;
	int	next_unused;
	struct passwd *pw;
	Authctxt *authctxt;
	pid_t	pid;
	int	forced;

	/* tty */
	char	*term;
	int	ptyfd, ttyfd, ptymaster;
	u_int	row, col, xpixel, ypixel;
	char	tty[TTYSZ];

	/* X11 */
	u_int	display_number;
	char	*display;
	u_int	screen;
	char	*auth_display;
	char	*auth_proto;
	char	*auth_data;
	int	single_connection;

	int	chanid;
	int	*x11_chanids;
	int	is_subsystem;
	char	*subsys;
	u_int	num_env;
	struct {
		char	*name;
		char	*val;
	} *env;
};

void	 do_authenticated(struct ssh *, Authctxt *);
void	 do_cleanup(struct ssh *, Authctxt *);

int	 session_open(Authctxt *, int);
void	 session_unused(int);
int	 session_input_channel_req(struct ssh *, Channel *, const char *);
void	 session_close_by_pid(struct ssh *ssh, pid_t, int);
void	 session_close_by_channel(struct ssh *, int, int, void *);
void	 session_destroy_all(struct ssh *, void (*)(Session *));
void	 session_pty_cleanup2(Session *);

Session	*session_new(void);
Session	*session_by_tty(char *);
void	 session_close(struct ssh *, Session *);
void	 do_setusercontext(struct passwd *);

const char	*session_get_remote_name_or_ip(struct ssh *, u_int, int);

#endif
