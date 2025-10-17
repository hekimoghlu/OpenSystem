/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
 * Telnet server variable declarations
 */
extern char	options[256];
extern char	do_dont_resp[256];
extern char	will_wont_resp[256];
extern int	linemode;	/* linemode on/off */
#ifdef	LINEMODE
extern int	uselinemode;	/* what linemode to use (on/off) */
extern int	editmode;	/* edit modes in use */
extern int	useeditmode;	/* edit modes to use */
extern int	alwayslinemode;	/* command line option */
extern int	lmodetype;	/* Client support for linemode */
#endif	/* LINEMODE */
extern int	flowmode;	/* current flow control state */
extern int	restartany;	/* restart output on any character state */
#ifdef DIAGNOSTICS
extern int	diagnostic;	/* telnet diagnostic capabilities */
#endif /* DIAGNOSTICS */
#ifdef BFTPDAEMON
extern int	bftpd;		/* behave as bftp daemon */
#endif /* BFTPDAEMON */
#ifdef	AUTHENTICATION
extern int	auth_level;
#endif

extern slcfun	slctab[NSLC + 1];	/* slc mapping table */

extern char	*terminaltype;

/*
 * I/O data buffers, pointers, and counters.
 */
extern char	ptyobuf[BUFSIZ+NETSLOP], *pfrontp, *pbackp;

extern char	netibuf[BUFSIZ], *netip;

extern char	netobuf[BUFSIZ], *nfrontp, *nbackp;
extern char	*neturg;		/* one past last bye of urgent data */

extern int	pcc, ncc;

extern int	mpty, spty, net;
extern char	line[16];
extern int	SYNCHing;		/* we are in TELNET SYNCH mode */

extern void
	_termstat(void),
	add_slc(char, char, cc_t),
	check_slc(void),
	change_slc(char, char, cc_t),
	cleanup(int),
	clientstat(int, int, int),
	copy_termbuf(char *, size_t),
	deferslc(void),
	defer_terminit(void),
	do_opt_slc(unsigned char *, int),
	doeof(void),
	dooption(int),
	dontoption(int),
	edithost(char *, char *),
	fatal(int, const char *),
	fatalperror(int, const char *),
	get_slc_defaults(void),
	init_env(void),
	init_termbuf(void),
	interrupt(void),
	localstat(void),
	flowstat(void),
	netclear(void),
	netflush(void),
#ifdef DIAGNOSTICS
	printoption(const char *, int),
	printdata(const char *, char *, int),
	printsub(char, unsigned char *, int),
#endif
	process_slc(unsigned char, unsigned char, cc_t),
	ptyflush(void),
	putchr(int),
	putf(char *, char *),
	recv_ayt(void),
	send_do(int, int),
	send_dont(int, int),
	send_slc(void),
	send_status(void),
	send_will(int, int),
	send_wont(int, int),
	sendbrk(void),
	sendsusp(void),
	set_termbuf(void),
	start_login(char *, int, char *),
	start_slc(int),
#ifdef	AUTHENTICATION
	start_slave(char *),
#else
	start_slave(char *, int, char *),
#endif
	suboption(void),
	telrcv(void),
	ttloop(void),
	tty_binaryin(int),
	tty_binaryout(int);

extern int
	end_slc(unsigned char **),
	getnpty(void),
#ifndef convex
	getpty(int *, int *),
#endif
	login_tty(int),
	spcset(int, cc_t *, cc_t **),
	stilloob(int),
	terminit(void),
	termstat(void),
	tty_flowmode(void),
	tty_restartany(void),
	tty_isbinaryin(void),
	tty_isbinaryout(void),
	tty_iscrnl(void),
	tty_isecho(void),
	tty_isediting(void),
	tty_islitecho(void),
	tty_isnewmap(void),
	tty_israw(void),
	tty_issofttab(void),
	tty_istrapsig(void),
	tty_linemode(void);

extern void
	tty_rspeed(int),
	tty_setecho(int),
	tty_setedit(int),
	tty_setlinemode(int),
	tty_setlitecho(int),
	tty_setsig(int),
	tty_setsofttab(int),
	tty_tspeed(int),
	willoption(int),
	wontoption(int);

int	output_data(const char *, ...) __printflike(1, 2);
void	output_datalen(const char *, int);
void	startslave(char *, int, char *);

#ifdef	ENCRYPTION
extern void	(*encrypt_output)(unsigned char *, int);
extern int	(*decrypt_input)(int);
extern char	*nclearto;
#endif	/* ENCRYPTION */


/*
 * The following are some clocks used to decide how to interpret
 * the relationship between various variables.
 */

extern struct {
    int
	system,			/* what the current time is */
	echotoggle,		/* last time user entered echo character */
	modenegotiated,		/* last time operating mode negotiated */
	didnetreceive,		/* last time we read data from network */
	ttypesubopt,		/* ttype subopt is received */
	tspeedsubopt,		/* tspeed subopt is received */
	environsubopt,		/* environ subopt is received */
	oenvironsubopt,		/* old environ subopt is received */
	xdisplocsubopt,		/* xdisploc subopt is received */
	baseline,		/* time started to do timed action */
	gotDM;			/* when did we last see a data mark */
} clocks;

#ifndef	DEFAULT_IM
#ifdef __APPLE__
#     define DEFAULT_IM  "\r\n\r\nFreeBSD (%h) (%t)\r\n\r\r\n\r"
#else
#   ifdef ultrix
#    define DEFAULT_IM	"\r\n\r\nULTRIX (%h) (%t)\r\n\r\r\n\r"
#   else
#    ifdef __FreeBSD__
#     define DEFAULT_IM  "\r\n\r\nFreeBSD (%h) (%t)\r\n\r\r\n\r"
#    else
#    define DEFAULT_IM	"\r\n\r\n4.4 BSD UNIX (%h) (%t)\r\n\r\r\n\r"
#    endif
#   endif
#endif /* __APPLE__ */
#endif
