/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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
/* System library. */

#include <sys_defs.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <syslog.h>
#include <signal.h>
#include <time.h>
#include <ctype.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <vstream.h>
#include <vstring_vstream.h>
#include <get_hostname.h>
#include <listen.h>
#include <events.h>
#include <mymalloc.h>
#include <iostuff.h>
#include <msg_vstream.h>
#include <stringops.h>
#include <sane_accept.h>
#include <inet_proto.h>
#include <myaddrinfo.h>
#include <make_dirs.h>
#include <myrand.h>
#include <chroot_uid.h>

/* Global library. */

#include <smtp_stream.h>
#include <mail_date.h>
#include <mail_version.h>

/* Application-specific. */

typedef struct SINK_STATE {
    VSTREAM *stream;
    VSTRING *buffer;
    int     data_state;
    int     (*read_fn) (struct SINK_STATE *);
    int     in_mail;
    int     rcpts;
    char   *push_back_ptr;
    /* Capture file information for fake Received: header */
    MAI_HOSTADDR_STR client_addr;	/* IP address */
    char   *addr_prefix;		/* ipv6: or empty */
    char   *helo_args;			/* text after HELO or EHLO */
    const char *client_proto;		/* SMTP, ESMTP, LMTP */
    time_t  start_time;			/* MAIL command time */
    int     id;				/* pseudo-random */
    VSTREAM *dump_file;			/* dump file or null */
    void    (*delayed_response) (struct SINK_STATE *state, const char *);
    char   *delayed_args;
} SINK_STATE;

#define ST_ANY			0
#define ST_CR			1
#define ST_CR_LF		2
#define ST_CR_LF_DOT		3
#define ST_CR_LF_DOT_CR		4
#define ST_CR_LF_DOT_CR_LF	5

#define PUSH_BACK_PEEK(state)		(*(state)->push_back_ptr != 0)
#define PUSH_BACK_GET(state)		(*(state)->push_back_ptr++)
#define PUSH_BACK_SET(state, text)	((state)->push_back_ptr = (text))

#ifndef DEF_MAX_CLIENT_COUNT
#define DEF_MAX_CLIENT_COUNT	256
#endif

#define SOFT_ERROR_RESP		"450 4.3.0 Error: command failed"
#define HARD_ERROR_RESP		"500 5.3.0 Error: command failed"

 /*
  * We can't rely on vstream auto-flushing, so we have to prepare for the
  * next read request.
  */
#define SMTP_FLUSH(fp) do { \
    if (vstream_peek(fp) <= 0 && readable(vstream_fileno(fp)) <= 0) \
        smtp_flush(fp); \
    } while (0)

static int var_tmout = 100;
static int var_max_line_length = 2048;
static char *var_myhostname;
static char *soft_error_resp = SOFT_ERROR_RESP;
static char *hard_error_resp = HARD_ERROR_RESP;
static int command_read(SINK_STATE *);
static int data_read(SINK_STATE *);
static void disconnect(SINK_STATE *);
static void read_timeout(int, void *);
static void read_event(int, void *);
static int show_count;
static int sess_count;
static int quit_count;
static int mesg_count;
static int max_quit_count;
static int max_msg_quit_count;
static int disable_pipelining;
static int disable_8bitmime;
static int disable_esmtp;
static int enable_lmtp;
static int pretend_pix;
static int disable_saslauth;
static int disable_xclient;
static int disable_xforward;
static int disable_enh_status;
static int disable_dsn;
static int max_client_count = DEF_MAX_CLIENT_COUNT;
static int client_count;
static int sock;
static int abort_delay = -1;
static int data_read_delay = 0;

static char *single_template;		/* individual template */
static char *shared_template;		/* shared template */
static VSTRING *start_string;		/* dump content prefix */

static INET_PROTO_INFO *proto_info;

#define STR(x)	vstring_str(x)

/* do_stats - show counters */

static void do_stats(void)
{
    vstream_printf("sess=%d quit=%d mesg=%d\r",
		   sess_count, quit_count, mesg_count);
    vstream_fflush(VSTREAM_OUT);
}

/* hard_err_resp - generic hard error response */

static void hard_err_resp(SINK_STATE *state)
{
    smtp_printf(state->stream, "%s", hard_error_resp);
    SMTP_FLUSH(state->stream);
}

/* soft_err_resp - generic soft error response */

static void soft_err_resp(SINK_STATE *state)
{
    smtp_printf(state->stream, "%s", soft_error_resp);
    SMTP_FLUSH(state->stream);
}

/* exp_path_template - expand template pathname, static result */

static VSTRING *exp_path_template(const char *template, time_t start_time)
{
    static VSTRING *path_buf = 0;
    struct tm *lt;

    if (path_buf == 0)
	path_buf = vstring_alloc(100);
    else
	VSTRING_RESET(path_buf);
    lt = localtime(&start_time);
    while (strftime(STR(path_buf), vstring_avail(path_buf), template, lt) == 0)
	VSTRING_SPACE(path_buf, vstring_avail(path_buf) + 100);
    VSTRING_SKIP(path_buf);
    return (path_buf);
}

/* make_parent_dir - create parent directory or bust */

static void make_parent_dir(const char *path, mode_t mode)
{
    const char *parent;

    parent = sane_dirname((VSTRING *) 0, path);
    if (make_dirs(parent, mode) < 0)
	msg_fatal("mkdir %s: %m", parent);
}

/* mail_file_open - open mail capture file */

static void mail_file_open(SINK_STATE *state)
{
    const char *myname = "mail_file_open";
    VSTRING *path_buf;
    ssize_t len;
    int     tries = 0;

    /*
     * Save the start time for later.
     */
    time(&(state->start_time));

    /*
     * Expand the per-message dumpfile pathname template.
     */
    path_buf = exp_path_template(single_template, state->start_time);

    /*
     * Append a random hexadecimal string to the pathname and create a new
     * file. Retry with a different path if the file already exists. Create
     * intermediate directories on the fly when the template specifies
     * multiple pathname segments.
     */
#define ID_FORMAT	"%08x"

    for (len = VSTRING_LEN(path_buf); /* void */ ; vstring_truncate(path_buf, len)) {
	if (++tries > 100)
	    msg_fatal("%s: something is looping", myname);
	state->id = myrand();
	vstring_sprintf_append(path_buf, ID_FORMAT, state->id);
	if ((state->dump_file = vstream_fopen(STR(path_buf),
					      O_RDWR | O_CREAT | O_EXCL,
					      0644)) != 0) {
	    break;
	} else if (errno == EEXIST) {
	    continue;
	} else if (errno == ENOENT) {
	    make_parent_dir(STR(path_buf), 0755);
	    continue;
	} else {
	    msg_fatal("open %s: %m", STR(path_buf));
	}
    }

    /*
     * Don't leave temporary files behind.
     */
    if (shared_template != 0 && unlink(STR(path_buf)) < 0)
	msg_fatal("unlink %s: %m", STR(path_buf));

    /*
     * Do initial header records.
     */
    if (start_string)
	vstream_fprintf(state->dump_file, "%s", STR(start_string));
    vstream_fprintf(state->dump_file, "X-Client-Addr: %s%s\n",
		    state->addr_prefix, state->client_addr.buf);
    vstream_fprintf(state->dump_file, "X-Client-Proto: %s\n", state->client_proto);
    if (state->helo_args)
	vstream_fprintf(state->dump_file, "X-Helo-Args: %s\n", state->helo_args);
    /* Note: there may be more than one recipient. */
}

/* mail_file_finish_header - do final smtp-sink generated header records */

static void mail_file_finish_header(SINK_STATE *state)
{
    if (state->helo_args)
	vstream_fprintf(state->dump_file, "Received: from %s ([%s%s])\n",
			state->helo_args, state->addr_prefix,
			state->client_addr.buf);
    else
	vstream_fprintf(state->dump_file, "Received: from [%s%s] ([%s%s])\n",
			state->addr_prefix, state->client_addr.buf,
			state->addr_prefix, state->client_addr.buf);
    vstream_fprintf(state->dump_file, "\tby %s (smtp-sink)"
		    " with %s id " ID_FORMAT ";\n",
		    var_myhostname, state->client_proto, state->id);
    vstream_fprintf(state->dump_file, "\t%s\n", mail_date(state->start_time));
}

/* mail_file_cleanup - common cleanup for capture file */

static void mail_file_cleanup(SINK_STATE *state)
{
    (void) vstream_fclose(state->dump_file);
    state->dump_file = 0;
}

/* mail_file_finish - handle message completion for capture file */

static void mail_file_finish(SINK_STATE *state)
{

    /*
     * Optionally append the captured message to a shared dumpfile.
     */
    if (shared_template) {
	const char *out_path;
	VSTREAM *out_fp;
	ssize_t count;

	/*
	 * Expand the shared dumpfile pathname template.
	 */
	out_path = STR(exp_path_template(shared_template, state->start_time));

	/*
	 * Open the shared dump file.
	 */
#define OUT_OPEN_FLAGS	(O_WRONLY | O_CREAT | O_APPEND)
#define OUT_OPEN_MODE	0644

	if ((out_fp = vstream_fopen(out_path, OUT_OPEN_FLAGS, OUT_OPEN_MODE))
	    == 0 && errno == ENOENT) {
	    make_parent_dir(out_path, 0755);
	    out_fp = vstream_fopen(out_path, OUT_OPEN_FLAGS, OUT_OPEN_MODE);
	}
	if (out_fp == 0)
	    msg_fatal("open %s: %m", out_path);

	/*
	 * Append message content from single-message dump file.
	 */
	if (vstream_fseek(state->dump_file, 0L, SEEK_SET) < 0)
	    msg_fatal("seek file %s: %m", VSTREAM_PATH(state->dump_file));
	VSTRING_RESET(state->buffer);
	for (;;) {
	    count = vstream_fread(state->dump_file, STR(state->buffer),
				  vstring_avail(state->buffer));
	    if (count <= 0)
		break;
	    if (vstream_fwrite(out_fp, STR(state->buffer), count) != count)
		msg_fatal("append file %s: %m", out_path);
	}
	if (vstream_ferror(state->dump_file))
	    msg_fatal("read file %s: %m", VSTREAM_PATH(state->dump_file));
	if (vstream_fclose(out_fp))
	    msg_fatal("append file %s: %m", out_path);
    }
    mail_file_cleanup(state);
}

/* mail_file_reset - abort mail to capture file */

static void mail_file_reset(SINK_STATE *state)
{
    if (shared_template == 0
	&& unlink(VSTREAM_PATH(state->dump_file)) < 0
	&& errno != ENOENT)
	msg_fatal("unlink %s: %m", VSTREAM_PATH(state->dump_file));
    mail_file_cleanup(state);
}

/* mail_cmd_reset - reset mail transaction information */

static void mail_cmd_reset(SINK_STATE *state)
{
    state->in_mail = 0;
    /* Not: state->rcpts = 0. This breaks the DOT reply with LMTP. */
    if (state->dump_file)
	mail_file_reset(state);
}

/* ehlo_response - respond to EHLO command */

static void ehlo_response(SINK_STATE *state, const char *args)
{
#define SKIP(cp, cond) do { \
	for (/* void */; *cp && (cond); cp++) \
	    /* void */; \
    } while (0)

    /* EHLO aborts a mail transaction in progress. */
    mail_cmd_reset(state);
    if (enable_lmtp == 0)
	state->client_proto = "ESMTP";
    smtp_printf(state->stream, "250-%s", var_myhostname);
    if (!disable_pipelining)
	smtp_printf(state->stream, "250-PIPELINING");
    if (!disable_8bitmime)
	smtp_printf(state->stream, "250-8BITMIME");
    if (!disable_saslauth)
	smtp_printf(state->stream, "250-AUTH PLAIN LOGIN");
    if (!disable_xclient)
	smtp_printf(state->stream, "250-XCLIENT NAME HELO");
    if (!disable_xforward)
	smtp_printf(state->stream, "250-XFORWARD NAME ADDR PROTO HELO");
    if (!disable_enh_status)
	smtp_printf(state->stream, "250-ENHANCEDSTATUSCODES");
    if (!disable_dsn)
	smtp_printf(state->stream, "250-DSN");
    /* RFC 821/2821/5321: Format is replycode<SPACE>optional-text<CRLF> */
    smtp_printf(state->stream, "250 ");
    SMTP_FLUSH(state->stream);
    if (single_template) {
	if (state->helo_args)
	    myfree(state->helo_args);
	SKIP(args, ISSPACE(*args));
	state->helo_args = mystrdup(args);
    }
}

/* helo_response - respond to HELO command */

static void helo_response(SINK_STATE *state, const char *args)
{
    /* HELO aborts a mail transaction in progress. */
    mail_cmd_reset(state);
    state->client_proto = "SMTP";
    smtp_printf(state->stream, "250 %s", var_myhostname);
    SMTP_FLUSH(state->stream);
    if (single_template) {
	if (state->helo_args)
	    myfree(state->helo_args);
	SKIP(args, ISSPACE(*args));
	state->helo_args = mystrdup(args);
    }
}

/* ok_response - send 250 OK */

static void ok_response(SINK_STATE *state, const char *unused_args)
{
    smtp_printf(state->stream, "250 2.0.0 Ok");
    SMTP_FLUSH(state->stream);
}

/* rset_response - reset, send 250 OK */

static void rset_response(SINK_STATE *state, const char *unused_args)
{
    mail_cmd_reset(state);
    smtp_printf(state->stream, "250 2.1.0 Ok");
    SMTP_FLUSH(state->stream);
}

/* mail_response - reset recipient count, send 250 OK */

static void mail_response(SINK_STATE *state, const char *args)
{
    if (state->in_mail) {
	smtp_printf(state->stream, "503 5.5.1 Error: nested MAIL command");
	SMTP_FLUSH(state->stream);
	return;
    }
    state->in_mail++;
    state->rcpts = 0;
    smtp_printf(state->stream, "250 2.1.0 Ok");
    SMTP_FLUSH(state->stream);
    if (single_template) {
	mail_file_open(state);
	SKIP(args, *args != ':');
	SKIP(args, *args == ':');
	SKIP(args, ISSPACE(*args));
	vstream_fprintf(state->dump_file, "X-Mail-Args: %s\n", args);
    }
}

/* rcpt_response - bump recipient count, send 250 OK */

static void rcpt_response(SINK_STATE *state, const char *args)
{
    if (state->in_mail == 0) {
	smtp_printf(state->stream, "503 5.5.1 Error: need MAIL command");
	SMTP_FLUSH(state->stream);
	return;
    }
    state->rcpts++;
    smtp_printf(state->stream, "250 2.1.5 Ok");
    SMTP_FLUSH(state->stream);
    /* Note: there may be more than one recipient per mail transaction. */
    if (state->dump_file) {
	SKIP(args, *args != ':');
	SKIP(args, *args == ':');
	SKIP(args, ISSPACE(*args));
	vstream_fprintf(state->dump_file, "X-Rcpt-Args: %s\n", args);
    }
}

/* abort_event - delayed abort after DATA command */

static void abort_event(int unused_event, void *context)
{
    SINK_STATE *state = (SINK_STATE *) context;

    smtp_printf(state->stream, "550 This violates SMTP");
    SMTP_FLUSH(state->stream);
    disconnect(state);
}

/* delay_read_event - resume input event handling */

static void delay_read_event(int event, void *context)
{
    SINK_STATE *state = (SINK_STATE *) context;

    if (event != EVENT_TIME)
	msg_panic("delay_read_event: non-timer event %d", event);

    event_enable_read(vstream_fileno(state->stream), read_event, (void *) state);
    event_request_timer(read_timeout, (void *) state, var_tmout);
}

/* delay_read - temporarily suspend input event handling */

static void delay_read(SINK_STATE *state, int delay)
{
    event_disable_readwrite(vstream_fileno(state->stream));
    event_cancel_timer(read_timeout, (void *) state);
    event_request_timer(delay_read_event, (void *) state, delay);
}

/* data_response - respond to DATA command */

static void data_response(SINK_STATE *state, const char *unused_args)
{
    if (state->in_mail == 0 || state->rcpts == 0) {
	smtp_printf(state->stream, "503 5.5.1 Error: need RCPT command");
	SMTP_FLUSH(state->stream);
	return;
    }
    /* Not: ST_ANY. */
    state->data_state = ST_CR_LF;
    smtp_printf(state->stream, "354 End data with <CR><LF>.<CR><LF>");
    SMTP_FLUSH(state->stream);
    if (abort_delay < 0) {
	state->read_fn = data_read;
	/* Todo: move into code that invokes the command response function. */
	if (data_read_delay > 0)
	    delay_read(state, data_read_delay);
    } else {
	/* Stop reading, send premature 550, and disconnect. */
	event_disable_readwrite(vstream_fileno(state->stream));
	event_cancel_timer(read_event, (void *) state);
	event_request_timer(abort_event, (void *) state, abort_delay);
    }
    if (state->dump_file)
	mail_file_finish_header(state);
}

/* dot_resp_hard - hard error response to . command */

static void dot_resp_hard(SINK_STATE *state)
{
    if (enable_lmtp) {
	while (state->rcpts-- > 0)	/* XXX this could block */
	    smtp_printf(state->stream, "%s", hard_error_resp);
    } else {
	smtp_printf(state->stream, "%s", hard_error_resp);
    }
    SMTP_FLUSH(state->stream);
}

/* dot_resp_soft - soft error response to . command */

static void dot_resp_soft(SINK_STATE *state)
{
    if (enable_lmtp) {
	while (state->rcpts-- > 0)	/* XXX this could block */
	    smtp_printf(state->stream, "%s", soft_error_resp);
    } else {
	smtp_printf(state->stream, "%s", soft_error_resp);
    }
    SMTP_FLUSH(state->stream);
}

/* dot_response - response to . command */

static void dot_response(SINK_STATE *state, const char *unused_args)
{
    if (enable_lmtp) {
	while (state->rcpts-- > 0)	/* XXX this could block */
	    smtp_printf(state->stream, "250 2.2.0 Ok");
    } else {
	smtp_printf(state->stream, "250 2.0.0 Ok");
    }
    SMTP_FLUSH(state->stream);
}

/* quit_response - respond to QUIT command */

static void quit_response(SINK_STATE *state, const char *unused_args)
{
    smtp_printf(state->stream, "221 Bye");
    smtp_flush(state->stream);			/* not: SMTP_FLUSH */
    if (show_count)
	quit_count++;
}

/* conn_response - respond to connect command */

static void conn_response(SINK_STATE *state, const char *unused_args)
{
    if (pretend_pix)
	smtp_printf(state->stream, "220 ********");
    else if (disable_esmtp)
	smtp_printf(state->stream, "220 %s", var_myhostname);
    else
	smtp_printf(state->stream, "220 %s ESMTP", var_myhostname);
    SMTP_FLUSH(state->stream);
}

/* delay_event - delayed command response */

static void delay_event(int unused_event, void *context)
{
    SINK_STATE *state = (SINK_STATE *) context;

    switch (vstream_setjmp(state->stream)) {

    default:
	msg_panic("unknown read/write error");
	/* NOTREACHED */

    case SMTP_ERR_TIME:
	msg_warn("write timeout");
	disconnect(state);
	return;

    case SMTP_ERR_EOF:
	msg_warn("lost connection");
	disconnect(state);
	return;

    case 0:
	state->delayed_response(state, state->delayed_args);
	myfree(state->delayed_args);
	state->delayed_args = 0;
	break;
    }

    if (state->delayed_response == quit_response) {
	disconnect(state);
	return;
    }
    state->delayed_response = 0;

    /* Resume input event handling after the delayed response. */
    event_enable_read(vstream_fileno(state->stream), read_event, (void *) state);
    event_request_timer(read_timeout, (void *) state, var_tmout);
}

/* data_read - read data from socket */

static int data_read(SINK_STATE *state)
{
    int     ch;
    struct data_trans {
	int     state;
	int     want;
	int     next_state;
    };
    static struct data_trans data_trans[] = {
	ST_ANY, '\r', ST_CR,
	ST_CR, '\n', ST_CR_LF,
	ST_CR_LF, '.', ST_CR_LF_DOT,
	ST_CR_LF_DOT, '\r', ST_CR_LF_DOT_CR,
	ST_CR_LF_DOT_CR, '\n', ST_CR_LF_DOT_CR_LF,
    };
    struct data_trans *dp;

    /*
     * A read may result in EOF, but is never supposed to time out - a time
     * out means that we were trying to read when no data was available.
     */
    for (;;) {
	if ((ch = VSTREAM_GETC(state->stream)) == VSTREAM_EOF)
	    return (-1);
	for (dp = data_trans; dp->state != state->data_state; dp++)
	     /* void */ ;

	/*
	 * Try to match the current character desired by the state machine.
	 * If that fails, try to restart the machine with a match for its
	 * first state.  This covers the case of a CR/LF/CR/LF sequence
	 * (empty line) right before the end of the message data.
	 */
	if (ch == dp->want)
	    state->data_state = dp->next_state;
	else if (ch == data_trans[0].want)
	    state->data_state = data_trans[0].next_state;
	else
	    state->data_state = ST_ANY;
	if (state->dump_file) {
	    if (ch != '\r' && state->data_state != ST_CR_LF_DOT)
		VSTREAM_PUTC(ch, state->dump_file);
	    if (vstream_ferror(state->dump_file))
		msg_fatal("append file %s: %m", VSTREAM_PATH(state->dump_file));
	}
	if (state->data_state == ST_CR_LF_DOT_CR_LF) {
	    PUSH_BACK_SET(state, ".\r\n");
	    state->read_fn = command_read;
	    state->data_state = ST_ANY;
	    if (state->dump_file)
		mail_file_finish(state);
	    mail_cmd_reset(state);
	    if (show_count || max_msg_quit_count > 0) {
		mesg_count++;
		if (show_count)
		    do_stats();
		if (max_msg_quit_count > 0 && mesg_count >= max_msg_quit_count)
		    exit(0);
	    }
	    break;
	}

	/*
	 * We must avoid blocking I/O, so get out of here as soon as both the
	 * VSTREAM and kernel read buffers dry up.
	 */
	if (vstream_peek(state->stream) <= 0
	    && readable(vstream_fileno(state->stream)) <= 0)
	    return (0);
    }
    return (0);
}

 /*
  * The table of all SMTP commands that we can handle.
  */
typedef struct SINK_COMMAND {
    const char *name;
    void    (*response) (SINK_STATE *, const char *);
    void    (*hard_response) (SINK_STATE *);
    void    (*soft_response) (SINK_STATE *);
    int     flags;
    int     delay;
    int     delay_odds;
} SINK_COMMAND;

#define FLAG_ENABLE	(1<<0)		/* command is enabled */
#define FLAG_SYSLOG	(1<<1)		/* log the command */
#define FLAG_HARD_ERR	(1<<2)		/* report hard error */
#define FLAG_SOFT_ERR	(1<<3)		/* report soft error */
#define FLAG_DISCONNECT	(1<<4)		/* disconnect */
#define FLAG_CLOSE	(1<<5)		/* say goodbye and disconnect */

static SINK_COMMAND command_table[] = {
    "connect", conn_response, hard_err_resp, soft_err_resp, 0, 0, 0,
    "helo", helo_response, hard_err_resp, soft_err_resp, 0, 0, 0,
    "ehlo", ehlo_response, hard_err_resp, soft_err_resp, 0, 0, 0,
    "lhlo", ehlo_response, hard_err_resp, soft_err_resp, 0, 0, 0,
    "xclient", ok_response, hard_err_resp, soft_err_resp, FLAG_ENABLE, 0, 0,
    "xforward", ok_response, hard_err_resp, soft_err_resp, FLAG_ENABLE, 0, 0,
    "auth", ok_response, hard_err_resp, soft_err_resp, FLAG_ENABLE, 0, 0,
    "mail", mail_response, hard_err_resp, soft_err_resp, FLAG_ENABLE, 0, 0,
    "rcpt", rcpt_response, hard_err_resp, soft_err_resp, FLAG_ENABLE, 0, 0,
    "data", data_response, hard_err_resp, soft_err_resp, FLAG_ENABLE, 0, 0,
    ".", dot_response, dot_resp_hard, dot_resp_soft, FLAG_ENABLE, 0, 0,
    "rset", rset_response, hard_err_resp, soft_err_resp, FLAG_ENABLE, 0, 0,
    "noop", ok_response, hard_err_resp, soft_err_resp, FLAG_ENABLE, 0, 0,
    "vrfy", ok_response, hard_err_resp, soft_err_resp, FLAG_ENABLE, 0, 0,
    "quit", quit_response, hard_err_resp, soft_err_resp, FLAG_ENABLE, 0, 0,
    0,
};

/* reset_cmd_flags - reset per-command command flags */

static void reset_cmd_flags(const char *cmd, int flags)
{
    SINK_COMMAND *cmdp;

    for (cmdp = command_table; cmdp->name != 0; cmdp++)
	if (strcasecmp(cmd, cmdp->name) == 0)
	    break;
    if (cmdp->name == 0)
	msg_fatal("unknown command: %s", cmd);
    cmdp->flags &= ~flags;
}

/* set_cmd_flags - set per-command command flags */

static void set_cmd_flags(const char *cmd, int flags)
{
    SINK_COMMAND *cmdp;

    for (cmdp = command_table; cmdp->name != 0; cmdp++)
	if (strcasecmp(cmd, cmdp->name) == 0)
	    break;
    if (cmdp->name == 0)
	msg_fatal("unknown command: %s", cmd);
    cmdp->flags |= flags;
}

/* set_cmds_flags - set per-command flags for multiple commands */

static void set_cmds_flags(const char *cmds, int flags)
{
    char   *saved_cmds;
    char   *cp;
    char   *cmd;

    saved_cmds = cp = mystrdup(cmds);
    while ((cmd = mystrtok(&cp, CHARS_COMMA_SP)) != 0)
	set_cmd_flags(cmd, flags);
    myfree(saved_cmds);
}

/* set_cmd_delay - set per-command delay */

static void set_cmd_delay(const char *cmd, int delay, int odds)
{
    SINK_COMMAND *cmdp;

    for (cmdp = command_table; cmdp->name != 0; cmdp++)
	if (strcasecmp(cmd, cmdp->name) == 0)
	    break;
    if (cmdp->name == 0)
	msg_fatal("unknown command: %s", cmd);

    if (delay <= 0)
	msg_fatal("non-positive '%s' delay", cmd);
    if (odds < 0 || odds > 99)
	msg_fatal("delay odds for '%s' out of range", cmd);

    cmdp->delay = delay;
    cmdp->delay_odds = odds;
}

/* set_cmd_delay_arg - set per-command delay from option argument */

static void set_cmd_delay_arg(char *arg)
{
    char   *cp;
    char   *saved_arg;
    char   *cmd;
    char   *delay;
    char   *odds;

    saved_arg = cp = mystrdup(arg);
    cmd = mystrtok(&cp, ":");
    delay = mystrtok(&cp, ":");
    if (cmd == 0 || delay == 0)
	msg_fatal("invalid command delay argument: %s", arg);
    odds = mystrtok(&cp, "");
    set_cmd_delay(cmd, atoi(delay), odds ? atoi(odds) : 0);
    myfree(saved_arg);
}

/* command_resp - respond to command */

static int command_resp(SINK_STATE *state, SINK_COMMAND *cmdp,
			        const char *command, const char *args)
{
    /* We use raw syslog. Sanitize data content and length. */
    if (cmdp->flags & FLAG_SYSLOG)
	syslog(LOG_INFO, "%s %.100s", command, args);
    if (cmdp->flags & FLAG_DISCONNECT)
	return (-1);
    if (cmdp->flags & FLAG_CLOSE) {
	smtp_printf(state->stream, "421 4.0.0 Server closing connection");
	return (-1);
    }
    if (cmdp->flags & FLAG_HARD_ERR) {
	cmdp->hard_response(state);
	return (0);
    }
    if (cmdp->flags & FLAG_SOFT_ERR) {
	cmdp->soft_response(state);
	return (0);
    }
    if (cmdp->delay > 0) {
	int     delay = cmdp->delay;

	if (cmdp->delay_odds > 0)
	    for (delay = 0;
	     ((int) (100.0 * rand() / (RAND_MAX + 1.0))) < cmdp->delay_odds;
		 delay += cmdp->delay)
		 /* NOP */ ;
	/* Suspend input event handling while delaying the command response. */
	event_disable_readwrite(vstream_fileno(state->stream));
	event_cancel_timer(read_timeout, (void *) state);
	event_request_timer(delay_event, (void *) state, delay);
	state->delayed_response = cmdp->response;
	state->delayed_args = mystrdup(args);
    } else {
	cmdp->response(state, args);
	if (cmdp->response == quit_response)
	    return (-1);
    }
    return (0);
}

/* command_read - talk the SMTP protocol, server side */

static int command_read(SINK_STATE *state)
{
    char   *command;
    SINK_COMMAND *cmdp;
    int     ch;
    struct cmd_trans {
	int     state;
	int     want;
	int     next_state;
    };
    static struct cmd_trans cmd_trans[] = {
	ST_ANY, '\r', ST_CR,
	ST_CR, '\n', ST_CR_LF,
	0, 0, 0,
    };
    struct cmd_trans *cp;
    char   *ptr;

    /*
     * A read may result in EOF, but is never supposed to time out - a time
     * out means that we were trying to read when no data was available.
     */
#define NEXT_CHAR(state) \
    (PUSH_BACK_PEEK(state) ? PUSH_BACK_GET(state) : VSTREAM_GETC(state->stream))

    if (state->data_state == ST_CR_LF)
	state->data_state = ST_ANY;		/* XXX */
    for (;;) {
	if ((ch = NEXT_CHAR(state)) == VSTREAM_EOF)
	    return (-1);

	/*
	 * Sanity check. We don't want to store infinitely long commands.
	 */
	if (VSTRING_LEN(state->buffer) >= var_max_line_length) {
	    msg_warn("command line too long");
	    return (-1);
	}
	VSTRING_ADDCH(state->buffer, ch);

	/*
	 * Try to match the current character desired by the state machine.
	 * If that fails, try to restart the machine with a match for its
	 * first state.
	 */
	for (cp = cmd_trans; cp->state != state->data_state; cp++)
	    if (cp->want == 0)
		msg_panic("command_read: unknown state: %d", state->data_state);
	if (ch == cp->want)
	    state->data_state = cp->next_state;
	else if (ch == cmd_trans[0].want)
	    state->data_state = cmd_trans[0].next_state;
	else
	    state->data_state = ST_ANY;
	if (state->data_state == ST_CR_LF)
	    break;

	/*
	 * We must avoid blocking I/O, so get out of here as soon as both the
	 * VSTREAM and kernel read buffers dry up.
	 * 
	 * XXX Solaris non-blocking read() may fail on a socket when ioctl
	 * FIONREAD reports there is unread data. Diagnosis by Max Pashkov.
	 * As a workaround we use readable() (which uses poll or select())
	 * instead of peek_fd() (which uses ioctl FIONREAD). Workaround added
	 * 20020604.
	 */
	if (PUSH_BACK_PEEK(state) == 0 && vstream_peek(state->stream) <= 0
	    && readable(vstream_fileno(state->stream)) <= 0)
	    return (0);
    }

    /*
     * Properly terminate the result, and reset the buffer write pointer for
     * reading the next command. This is ugly, but not as ugly as trying to
     * deal with all the early returns below.
     */
    vstring_truncate(state->buffer, VSTRING_LEN(state->buffer) - 2);
    VSTRING_TERMINATE(state->buffer);
    state->data_state = ST_CR_LF;
    VSTRING_RESET(state->buffer);

    /*
     * Got a complete command line. Parse it.
     */
    ptr = vstring_str(state->buffer);
    if (msg_verbose)
	msg_info("%s", ptr);
    if ((command = mystrtok(&ptr, " \t")) == 0) {
	smtp_printf(state->stream, "500 5.5.2 Error: unknown command");
	SMTP_FLUSH(state->stream);
	return (0);
    }
    for (cmdp = command_table; cmdp->name != 0; cmdp++)
	if (strcasecmp(command, cmdp->name) == 0)
	    break;
    if (cmdp->name == 0 || (cmdp->flags & FLAG_ENABLE) == 0) {
	smtp_printf(state->stream, "500 5.5.1 Error: unknown command");
	SMTP_FLUSH(state->stream);
	return (0);
    }
    return (command_resp(state, cmdp, command, printable(ptr, '?')));
}

/* read_timeout - handle timer event */

static void read_timeout(int unused_event, void *context)
{
    SINK_STATE *state = (SINK_STATE *) context;

    /*
     * We don't send anything to the client, because we would have to set up
     * an smtp_stream exception handler first. And that is just too much
     * trouble.
     */
    msg_warn("read timeout");
    disconnect(state);
}

/* read_event - handle command or data read events */

static void read_event(int unused_event, void *context)
{
    SINK_STATE *state = (SINK_STATE *) context;

    /*
     * The input reading routine not only reads input (with vstream calls)
     * but also produces output (with smtp_stream calls). Because the output
     * routines can raise timeout or EOF exceptions with vstream_longjmp(),
     * the input reading routine needs to set up corresponding exception
     * handlers with vstream_setjmp(). Guarding the input operations in the
     * same manner is not useful: we must read input in non-blocking mode, so
     * we never get called when the socket stays unreadable too long. And EOF
     * is already trivial to detect with the vstream calls.
     */
    do {
	switch (vstream_setjmp(state->stream)) {

	default:
	    msg_panic("unknown read/write error");
	    /* NOTREACHED */

	case SMTP_ERR_TIME:
	    msg_warn("write timeout");
	    disconnect(state);
	    return;

	case SMTP_ERR_EOF:
	    msg_warn("lost connection");
	    disconnect(state);
	    return;

	case 0:
	    if (state->read_fn(state) < 0) {
		if (msg_verbose)
		    msg_info("disconnect");
		disconnect(state);
		return;
	    }
	}
    } while (PUSH_BACK_PEEK(state) != 0 || vstream_peek(state->stream) > 0);

    /*
     * Reset the idle timer. Wait until the next input event, or until the
     * idle timer goes off.
     */
    event_request_timer(read_timeout, (void *) state, var_tmout);
}

static void connect_event(int, void *);

/* disconnect - handle disconnection events */

static void disconnect(SINK_STATE *state)
{
    event_disable_readwrite(vstream_fileno(state->stream));
    event_cancel_timer(read_timeout, (void *) state);
    if (show_count) {
	sess_count++;
	do_stats();
    }
    vstream_fclose(state->stream);
    vstring_free(state->buffer);
    /* Clean up file capture attributes. */
    if (state->helo_args)
	myfree(state->helo_args);
    /* Delete incomplete mail transaction. */
    mail_cmd_reset(state);
    if (state->delayed_args)
	myfree(state->delayed_args);
    myfree((void *) state);
    if (max_quit_count > 0 && quit_count >= max_quit_count)
	exit(0);
    if (client_count-- == max_client_count)
	event_enable_read(sock, connect_event, (void *) 0);
}

/* connect_event - handle connection events */

static void connect_event(int unused_event, void *unused_context)
{
    struct sockaddr_storage ss;
    SOCKADDR_SIZE len = sizeof(ss);
    struct sockaddr *sa = (struct sockaddr *) &ss;
    SINK_STATE *state;
    int     fd;

    if ((fd = sane_accept(sock, sa, &len)) >= 0) {
	/* Safety: limit the number of open sockets and capture files. */
	if (++client_count == max_client_count)
	    event_disable_readwrite(sock);
	state = (SINK_STATE *) mymalloc(sizeof(*state));
	if (strchr((char *) proto_info->sa_family_list, sa->sa_family))
	    SOCKADDR_TO_HOSTADDR(sa, len, &state->client_addr,
				 (MAI_SERVPORT_STR *) 0, sa->sa_family);
	else
	    strncpy(state->client_addr.buf, "local", sizeof("local"));
	if (msg_verbose)
	    msg_info("connect (%s %s)",
#ifdef AF_LOCAL
		     sa->sa_family == AF_LOCAL ? "AF_LOCAL" :
#else
		     sa->sa_family == AF_UNIX ? "AF_UNIX" :
#endif
		     sa->sa_family == AF_INET ? "AF_INET" :
#ifdef AF_INET6
		     sa->sa_family == AF_INET6 ? "AF_INET6" :
#endif
		     "unknown protocol family",
		     state->client_addr.buf);
	non_blocking(fd, NON_BLOCKING);
	state->stream = vstream_fdopen(fd, O_RDWR);
	vstream_tweak_sock(state->stream);
	state->buffer = vstring_alloc(1024);
	state->read_fn = command_read;
	state->data_state = ST_ANY;
	PUSH_BACK_SET(state, "");
	smtp_timeout_setup(state->stream, var_tmout);
	state->in_mail = 0;
	state->rcpts = 0;
	state->delayed_response = 0;
	state->delayed_args = 0;
	/* Initialize file capture attributes. */
#ifdef AF_INET6
	if (sa->sa_family == AF_INET6)
	    state->addr_prefix = "ipv6:";
	else
#endif
	    state->addr_prefix = "";

	state->helo_args = 0;
	state->client_proto = enable_lmtp ? "LMTP" : "SMTP";
	state->start_time = 0;
	state->id = 0;
	state->dump_file = 0;

	/*
	 * We use the smtp_stream module to produce output. That module
	 * throws an exception via vstream_longjmp() in case of a timeout or
	 * lost connection error. Therefore we must prepare to handle these
	 * exceptions with vstream_setjmp().
	 */
	switch (vstream_setjmp(state->stream)) {

	default:
	    msg_panic("unknown read/write error");
	    /* NOTREACHED */

	case SMTP_ERR_TIME:
	    msg_warn("write timeout");
	    disconnect(state);
	    return;

	case SMTP_ERR_EOF:
	    msg_warn("lost connection");
	    disconnect(state);
	    return;

	case 0:
	    if (command_resp(state, command_table, "connect", "") < 0)
		disconnect(state);
	    else if (command_table->delay == 0) {
		event_enable_read(fd, read_event, (void *) state);
		event_request_timer(read_timeout, (void *) state, var_tmout);
	    }
	}
    }
}

/* usage - explain */

static void usage(char *myname)
{
    msg_fatal("usage: %s [-468acCeEFLpPv] [-A abort_delay] [-b soft_bounce_reply] [-B hard_bounce_reply] [-d dump-template] [-D dump-template] [-f commands] [-h hostname] [-m max_concurrency] [-M message_quit_count] [-n quit_count] [-q commands] [-r commands] [-R root-dir] [-s commands] [-S start-string] [-u user_privs] [-w delay] [host]:port backlog", myname);
}

MAIL_VERSION_STAMP_DECLARE;

int     main(int argc, char **argv)
{
    int     backlog;
    int     ch;
    int     delay;
    const char *protocols = INET_PROTO_NAME_ALL;
    const char *root_dir = 0;
    const char *user_privs = 0;

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    /*
     * Fix 20051207.
     */
    signal(SIGPIPE, SIG_IGN);

    /*
     * Initialize diagnostics.
     */
    msg_vstream_init(argv[0], VSTREAM_ERR);

    /*
     * Parse JCL.
     */
    while ((ch = GETOPT(argc, argv, "468aA:b:B:cCd:D:eEf:Fh:H:Ln:m:M:NpPq:Q:r:R:s:S:t:T:u:vw:W:")) > 0) {
	switch (ch) {
	case '4':
	    protocols = INET_PROTO_NAME_IPV4;
	    break;
	case '6':
	    protocols = INET_PROTO_NAME_IPV6;
	    break;
	case '8':
	    disable_8bitmime = 1;
	    break;
	case 'a':
	    disable_saslauth = 1;
	    break;
	case 'A':
	    if (!alldig(optarg) || (abort_delay = atoi(optarg)) < 0)
		usage(argv[0]);
	    break;
	case 'b':
	    if (optarg[0] != '4' || strspn(optarg, "0123456789") != 3) {
		msg_error("bad soft error reply: %s", optarg);
		usage(argv[0]);
	    } else
		soft_error_resp = optarg;
	    break;
	case 'B':
	    if (optarg[0] != '5' || strspn(optarg, "0123456789") != 3) {
		msg_error("bad hard error reply: %s", optarg);
		usage(argv[0]);
	    } else
		hard_error_resp = optarg;
	    break;
	case 'c':
	    show_count++;
	    break;
	case 'C':
	    disable_xclient = 1;
	    reset_cmd_flags("xclient", FLAG_ENABLE);
	    break;
	case 'd':
	    single_template = optarg;
	    break;
	case 'D':
	    shared_template = optarg;
	    break;
	case 'e':
	    disable_esmtp = 1;
	    break;
	case 'E':
	    disable_enh_status = 1;
	    break;
	case 'f':
	    set_cmds_flags(optarg, FLAG_HARD_ERR);
	    disable_pipelining = 1;
	    break;
	case 'F':
	    disable_xforward = 1;
	    reset_cmd_flags("xforward", FLAG_ENABLE);
	    break;
	case 'h':
	    var_myhostname = optarg;
	    break;
	case 'H':
	    if ((data_read_delay = atoi(optarg)) <= 0)
		msg_fatal("bad data read delay: %s", optarg);
	    break;
	case 'L':
	    enable_lmtp = 1;
	    break;
	case 'm':
	    if ((max_client_count = atoi(optarg)) <= 0)
		msg_fatal("bad concurrency limit: %s", optarg);
	    break;
	case 'M':
	    if ((max_msg_quit_count = atoi(optarg)) <= 0)
		msg_fatal("bad message quit count: %s", optarg);
	    break;
	case 'n':
	    if ((max_quit_count = atoi(optarg)) <= 0)
		msg_fatal("bad quit count: %s", optarg);
	    break;
	case 'N':
	    disable_dsn = 1;
	    break;
	case 'p':
	    disable_pipelining = 1;
	    break;
	case 'P':
	    pretend_pix = 1;
	    disable_esmtp = 1;
	    break;
	case 'q':
	    set_cmds_flags(optarg, FLAG_DISCONNECT);
	    break;
	case 'Q':
	    set_cmds_flags(optarg, FLAG_CLOSE);
	    break;
	case 'r':
	    set_cmds_flags(optarg, FLAG_SOFT_ERR);
	    disable_pipelining = 1;
	    break;
	case 'R':
	    root_dir = optarg;
	    break;
	case 's':
	    openlog(basename(argv[0]), LOG_PID, LOG_MAIL);
	    set_cmds_flags(optarg, FLAG_SYSLOG);
	    break;
	case 'S':
	    start_string = vstring_alloc(10);
	    unescape(start_string, optarg);
	    break;
	case 't':
	    if ((var_tmout = atoi(optarg)) <= 0)
		msg_fatal("bad timeout: %s", optarg);
	    break;
	case 'T':
	    if ((inet_windowsize = atoi(optarg)) <= 0)
		msg_fatal("bad TCP window size: %s", optarg);
	    break;
	case 'u':
	    user_privs = optarg;
	    break;
	case 'v':
	    msg_verbose++;
	    break;
	case 'w':
	    if ((delay = atoi(optarg)) <= 0)
		usage(argv[0]);
	    set_cmd_delay("data", delay, 0);
	    break;
	case 'W':
	    set_cmd_delay_arg(optarg);
	    break;
	default:
	    usage(argv[0]);
	}
    }
    if (argc - optind != 2)
	usage(argv[0]);
    if ((backlog = atoi(argv[optind + 1])) <= 0)
	usage(argv[0]);
    if (single_template && shared_template)
	msg_fatal("use only one of -d or -D, but not both");
    if (geteuid() == 0 && user_privs == 0)
	msg_fatal("-u option is required if running as root");

    /*
     * Initialize.
     */
    if (var_myhostname == 0)
	var_myhostname = "smtp-sink";
    set_cmds_flags(enable_lmtp ? "lhlo" :
		   disable_esmtp ? "helo" :
		   "helo, ehlo", FLAG_ENABLE);
    proto_info = inet_proto_init("protocols", protocols);
    if (strncmp(argv[optind], "unix:", 5) == 0) {
	sock = unix_listen(argv[optind] + 5, backlog, BLOCKING);
    } else {
	if (strncmp(argv[optind], "inet:", 5) == 0)
	    argv[optind] += 5;
	sock = inet_listen(argv[optind], backlog, BLOCKING);
    }
    if (user_privs)
	chroot_uid(root_dir, user_privs);

    if (single_template)
	mysrand((int) time((time_t *) 0));
    else if (shared_template)
	single_template = shared_template;

    /*
     * Start the event handler.
     */
    event_enable_read(sock, connect_event, (void *) 0);
    for (;;)
	event_loop(-1);
}
