/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
 * FTP global variables.
 */

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
#include <setjmp.h>

/*
 * Options and other state info.
 */
extern int	trace;			/* trace packets exchanged */
extern int	hash;			/* print # for each buffer transferred */
extern int	sendport;		/* use PORT cmd for each data connection */
extern int	verbose;		/* print messages coming back from server */
extern int	connected;		/* connected to server */
extern int	fromatty;		/* input is from a terminal */
extern int	interactive;		/* interactively prompt on m* cmds */
extern int	lineedit;		/* use line-editing */
extern int	debug;			/* debugging level */
extern int	bell;			/* ring bell on cmd completion */
extern int	doglob;			/* glob local file names */
extern int	autologin;		/* establish user account on connection */
extern int	doencrypt;
extern int	proxy;			/* proxy server connection active */
extern int	proxflag;		/* proxy connection exists */
extern int	sunique;		/* store files on server with unique name */
extern int	runique;		/* store local files with unique name */
extern int	mcase;			/* map upper to lower case for mget names */
extern int	ntflag;			/* use ntin ntout tables for name translation */
extern int	mapflag;		/* use mapin mapout templates on file names */
extern int	code;			/* return/reply code for ftp command */
extern int	crflag;			/* if 1, strip car. rets. on ascii gets */
extern char	pasv[64];		/* passive port for proxy data connection */
extern int	passivemode;		/* passive mode enabled */
extern char	*altarg;		/* argv[1] with no shell-like preprocessing  */
extern char	ntin[17];		/* input translation table */
extern char	ntout[17];		/* output translation table */
extern char	mapin[MaxPathLen];	/* input map template */
extern char	mapout[MaxPathLen];	/* output map template */
extern char	typename[32];		/* name of file transfer type */
extern int	type;			/* requested file transfer type */
extern int	curtype;		/* current file transfer type */
extern char	structname[32];		/* name of file transfer structure */
extern int	stru;			/* file transfer structure */
extern char	formname[32];		/* name of file transfer format */
extern int	form;			/* file transfer format */
extern char	modename[32];		/* name of file transfer mode */
extern int	mode;			/* file transfer mode */
extern char	bytename[32];		/* local byte size in ascii */
extern int	bytesize;		/* local byte size in binary */

extern char	*hostname;		/* name of host connected to */
extern int	unix_server;		/* server is unix, can use binary for ascii */
extern int	unix_proxy;		/* proxy is unix, can use binary for ascii */

extern jmp_buf	toplevel;		/* non-local goto stuff for cmd scanner */

extern char	line[200];		/* input line buffer */
extern char	*stringbase;		/* current scan point in line buffer */
extern char	argbuf[200];		/* argument storage buffer */
extern char	*argbase;		/* current storage point in arg buffer */
extern int	margc;			/* count of arguments on input line */
extern char	**margv;		/* args parsed from input line */
extern int	margvlen;		/* how large margv is currently */
extern int     cpend;                  /* flag: if != 0, then pending server reply */
extern int	mflag;			/* flag: if != 0, then active multi command */

extern int	options;		/* used during socket creation */
extern int      use_kerberos;           /* use Kerberos authentication */

/*
 * Format of command table.
 */
struct cmd {
	char	*c_name;	/* name of command */
	char	*c_help;	/* help string */
	char	c_bell;		/* give bell when command completes */
	char	c_conn;		/* must be connected to use command */
	char	c_proxy;	/* proxy server may execute */
	void	(*c_handler) (int, char **); /* function to call */
};

struct macel {
	char mac_name[9];	/* macro name */
	char *mac_start;	/* start of macro in macbuf */
	char *mac_end;		/* end of macro in macbuf */
};

extern int macnum;			/* number of defined macros */
extern struct macel macros[16];
extern char macbuf[4096];


