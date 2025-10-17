/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#ifdef __APPLE__
/* remove if / when Libc/include/protocols/talkd is updated */
#define tsockaddr osockaddr
#endif /* __APPLE__ */
#include <protocols/talkd.h>
#include <curses.h>
#include <signal.h>

extern	int sockt;
extern	int curses_initialized;
extern	int invitation_waiting;

extern	const char *current_state;
extern	int current_line;

extern volatile sig_atomic_t gotwinch;

typedef struct xwin {
	WINDOW	*x_win;
	int	x_nlines;
	int	x_ncols;
	int	x_line;
	int	x_col;
	char	kill;
	char	cerase;
	char	werase;
} xwin_t;

extern	xwin_t my_win;
extern	xwin_t his_win;
extern	WINDOW *line_win;

extern	void	announce_invite(void);
extern	int	check_local(void);
extern	void	check_writeable(void);
extern	void	ctl_transact(struct in_addr,CTL_MSG,int,CTL_RESPONSE *);
extern	void	disp_msg(int);
extern	void	end_msgs(void);
extern	void	get_addrs(const char *, const char *);
extern	int	get_iface(struct in_addr *, struct in_addr *);
extern	void	get_names(int, char **);
extern	void	init_display(void);
extern	void	invite_remote(void);
extern	int	look_for_invite(CTL_RESPONSE *);
extern	int	max(int, int);
extern	void	message(const char *);
extern	void	open_ctl(void);
extern	void	open_sockt(void);
extern	void	p_error(const char *);
extern	void	print_addr(struct sockaddr_in);
extern	void	quit(void);
extern	int	readwin(WINDOW *, int, int);
extern	void	re_invite(int);
extern	void	send_delete(void);
extern	void	set_edit_chars(void);
extern	void	sig_sent(int);
extern	void	sig_winch(int);
extern	void	start_msgs(void);
extern	void	talk(void);
extern	void	resize_display(void);
