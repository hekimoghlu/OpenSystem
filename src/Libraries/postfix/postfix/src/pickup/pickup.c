/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <ctype.h>

/* Utility library. */

#include <msg.h>
#include <scan_dir.h>
#include <vstring.h>
#include <vstream.h>
#include <set_ugid.h>
#include <safe_open.h>
#include <watchdog.h>
#include <stringops.h>

/* Global library. */

#include <mail_queue.h>
#include <mail_open_ok.h>
#include <mymalloc.h>
#include <mail_proto.h>
#include <cleanup_user.h>
#include <mail_date.h>
#include <mail_params.h>
#include <mail_conf.h>
#include <record.h>
#include <rec_type.h>
#include <lex_822.h>
#include <input_transp.h>
#include <rec_attr_map.h>
#include <mail_version.h>
#include <smtputf8.h>

/* Single-threaded server skeleton. */

#include <mail_server.h>

/* Application-specific. */

char   *var_filter_xport;
char   *var_input_transp;

 /*
  * Structure to bundle a bunch of information about a queue file.
  */
typedef struct {
    char   *id;				/* queue file basename */
    struct stat st;			/* queue file status */
    char   *path;			/* name for open/remove */
    char   *sender;			/* sender address */
} PICKUP_INFO;

 /*
  * What action should be taken after attempting to deliver a message: remove
  * the file from the maildrop, or leave it alone. The latter is also used
  * for files that are still being written to.
  */
#define REMOVE_MESSAGE_FILE	1
#define KEEP_MESSAGE_FILE	2

 /*
  * Transparency: before mail is queued, do we allow address mapping,
  * automatic bcc, header/body checks?
  */
int     pickup_input_transp_mask;

/* file_read_error - handle error while reading queue file */

static int file_read_error(PICKUP_INFO *info, int type)
{
    msg_warn("uid=%ld: unexpected or malformed record type %d",
	     (long) info->st.st_uid, type);
    return (REMOVE_MESSAGE_FILE);
}

/* cleanup_service_error_reason - handle error writing to cleanup service. */

static int cleanup_service_error_reason(PICKUP_INFO *info, int status,
					        const char *reason)
{

    /*
     * XXX If the cleanup server gave a reason, then it was already logged.
     * Don't bother logging it another time.
     * 
     * XXX Discard a message without recipient. This can happen with "postsuper
     * -r" when a message is already delivered (or bounced). The Postfix
     * sendmail command rejects submissions without recipients.
     */
    if (reason == 0 || *reason == 0)
	msg_warn("%s: error writing %s: %s",
		 info->path, info->id, cleanup_strerror(status));
    return ((status & (CLEANUP_STAT_BAD | CLEANUP_STAT_RCPT)) ?
	    REMOVE_MESSAGE_FILE : KEEP_MESSAGE_FILE);
}

#define cleanup_service_error(info, status) \
	cleanup_service_error_reason((info), (status), (char *) 0)

/* copy_segment - copy a record group */

static int copy_segment(VSTREAM *qfile, VSTREAM *cleanup, PICKUP_INFO *info,
			        VSTRING *buf, char *expected)
{
    int     type;
    int     check_first = (*expected == REC_TYPE_CONTENT[0]);
    int     time_seen = 0;
    char   *attr_name;
    char   *attr_value;
    char   *saved_attr;
    int     skip_attr;

    /*
     * Limit the input record size. All front-end programs should protect the
     * mail system against unreasonable inputs. This also requires that we
     * limit the size of envelope records written by the local posting agent.
     * 
     * Records with named attributes are filtered by postdrop(1).
     * 
     * We must allow PTR records here because of "postsuper -r".
     */
    for (;;) {
	if ((type = rec_get(qfile, buf, var_line_limit)) < 0
	    || strchr(expected, type) == 0)
	    return (file_read_error(info, type));
	if (msg_verbose)
	    msg_info("%s: read %c %s", info->id, type, vstring_str(buf));
	if (type == *expected)
	    break;
	if (type == REC_TYPE_FROM) {
	    if (info->sender == 0)
		info->sender = mystrdup(vstring_str(buf));
	    /* Compatibility with Postfix < 2.3. */
	    if (time_seen == 0)
		rec_fprintf(cleanup, REC_TYPE_TIME, "%ld",
			    (long) info->st.st_mtime);
	}
	if (type == REC_TYPE_TIME)
	    time_seen = 1;

	/*
	 * XXX Workaround: REC_TYPE_FILT (used in envelopes) == REC_TYPE_CONT
	 * (used in message content).
	 * 
	 * As documented in postsuper(1), ignore content filter record.
	 */
	if (*expected != REC_TYPE_CONTENT[0]) {
	    if (type == REC_TYPE_FILT)
		/* Discard FILTER record after "postsuper -r". */
		continue;
	    if (type == REC_TYPE_RDR)
		/* Discard REDIRECT record after "postsuper -r". */
		continue;
	}
	if (*expected == REC_TYPE_EXTRACT[0]) {
	    if (type == REC_TYPE_RRTO)
		/* Discard return-receipt record after "postsuper -r". */
		continue;
	    if (type == REC_TYPE_ERTO)
		/* Discard errors-to record after "postsuper -r". */
		continue;
	    if (type == REC_TYPE_ATTR) {
		saved_attr = mystrdup(vstring_str(buf));
		skip_attr = (split_nameval(saved_attr,
					   &attr_name, &attr_value) == 0
			     && rec_attr_map(attr_name) == 0);
		myfree(saved_attr);
		/* Discard other/header/body action after "postsuper -r". */
		if (skip_attr)
		    continue;
	    }
	}

	/*
	 * XXX Force an empty record when the queue file content begins with
	 * whitespace, so that it won't be considered as being part of our
	 * own Received: header. What an ugly Kluge.
	 */
	if (check_first
	    && (type == REC_TYPE_NORM || type == REC_TYPE_CONT)) {
	    check_first = 0;
	    if (VSTRING_LEN(buf) > 0 && IS_SPACE_TAB(vstring_str(buf)[0]))
		rec_put(cleanup, REC_TYPE_NORM, "", 0);
	}
	if ((REC_PUT_BUF(cleanup, type, buf)) < 0)
	    return (cleanup_service_error(info, CLEANUP_STAT_WRITE));
    }
    return (0);
}

/* pickup_copy - copy message to cleanup service */

static int pickup_copy(VSTREAM *qfile, VSTREAM *cleanup,
		               PICKUP_INFO *info, VSTRING *buf)
{
    time_t  now = time((time_t *) 0);
    int     status;
    char   *name;

    /*
     * Protect against time-warped time stamps. Warn about mail that has been
     * queued for an excessive amount of time. Allow for some time drift with
     * network clients that mount the maildrop remotely - especially clients
     * that can't get their daylight savings offsets right.
     */
#define DAY_SECONDS 86400
#define HOUR_SECONDS 3600

    if (info->st.st_mtime > now + 2 * HOUR_SECONDS) {
	msg_warn("%s: message dated %ld seconds into the future",
		 info->id, (long) (info->st.st_mtime - now));
	info->st.st_mtime = now;
    } else if (info->st.st_mtime < now - DAY_SECONDS) {
	msg_warn("%s: message has been queued for %d days",
		 info->id, (int) ((now - info->st.st_mtime) / DAY_SECONDS));
    }

    /*
     * Add content inspection transport. See also postsuper(1).
     */
    if (*var_filter_xport)
	rec_fprintf(cleanup, REC_TYPE_FILT, "%s", var_filter_xport);

    /*
     * Copy the message envelope segment. Allow only those records that we
     * expect to see in the envelope section. The envelope segment must
     * contain an envelope sender address.
     */
    if ((status = copy_segment(qfile, cleanup, info, buf, REC_TYPE_ENVELOPE)) != 0)
	return (status);
    if (info->sender == 0) {
	msg_warn("%s: uid=%ld: no envelope sender",
		 info->id, (long) info->st.st_uid);
	return (REMOVE_MESSAGE_FILE);
    }

    /*
     * For messages belonging to $mail_owner also log the maildrop queue id.
     * This supports message tracking for mail requeued via "postsuper -r".
     */
#define MAIL_IS_REQUEUED(info) \
    ((info)->st.st_uid == var_owner_uid && ((info)->st.st_mode & S_IROTH) == 0)

    if (MAIL_IS_REQUEUED(info)) {
	msg_info("%s: uid=%d from=<%s> orig_id=%s", info->id,
		 (int) info->st.st_uid, info->sender,
		 ((name = strrchr(info->path, '/')) != 0 ?
		  name + 1 : info->path));
    } else {
	msg_info("%s: uid=%d from=<%s>", info->id,
		 (int) info->st.st_uid, info->sender);
    }

    /*
     * Message content segment. Send a dummy message length. Prepend a
     * Received: header to the message contents. For tracing purposes,
     * include the message file ownership, without revealing the login name.
     */
    rec_fputs(cleanup, REC_TYPE_MESG, "");
    rec_fprintf(cleanup, REC_TYPE_NORM, "Received: by %s (%s, from userid %ld)",
		var_myhostname, var_mail_name, (long) info->st.st_uid);
    rec_fprintf(cleanup, REC_TYPE_NORM, "\tid %s; %s", info->id,
		mail_date(info->st.st_mtime));

    /*
     * Copy the message content segment. Allow only those records that we
     * expect to see in the message content section.
     */
    if ((status = copy_segment(qfile, cleanup, info, buf, REC_TYPE_CONTENT)) != 0)
	return (status);

    /*
     * Send the segment with information extracted from message headers.
     * Permit a non-empty extracted segment, so that list manager software
     * can to output recipients after the message, and so that sysadmins can
     * re-inject messages after a change of configuration.
     */
    rec_fputs(cleanup, REC_TYPE_XTRA, "");
    if ((status = copy_segment(qfile, cleanup, info, buf, REC_TYPE_EXTRACT)) != 0)
	return (status);

    /*
     * There are no errors. Send the end-of-data marker, and get the cleanup
     * service completion status. XXX Since the pickup service is unable to
     * bounce, the cleanup service can report only soft errors here.
     */
    rec_fputs(cleanup, REC_TYPE_END, "");
    if (attr_scan(cleanup, ATTR_FLAG_MISSING,
		  RECV_ATTR_INT(MAIL_ATTR_STATUS, &status),
		  RECV_ATTR_STR(MAIL_ATTR_WHY, buf),
		  ATTR_TYPE_END) != 2)
	return (cleanup_service_error(info, CLEANUP_STAT_WRITE));

    /*
     * Depending on the cleanup service completion status, delete the message
     * file, or try again later. Bounces are dealt with by the cleanup
     * service itself. The master process wakes up the cleanup service every
     * now and then.
     */
    if (status) {
	return (cleanup_service_error_reason(info, status, vstring_str(buf)));
    } else {
	return (REMOVE_MESSAGE_FILE);
    }
}

/* pickup_file - initialize for file copy and cleanup */

static int pickup_file(PICKUP_INFO *info)
{
    VSTRING *buf = vstring_alloc(100);
    int     status;
    VSTREAM *qfile;
    VSTREAM *cleanup;
    int     cleanup_flags;

    /*
     * Open the submitted file. If we cannot open it, and we're not having a
     * file descriptor leak problem, delete the submitted file, so that we
     * won't keep complaining about the same file again and again. XXX
     * Perhaps we should save "bad" files elsewhere for further inspection.
     * XXX How can we delete a file when open() fails with ENOENT?
     */
    qfile = safe_open(info->path, O_RDONLY | O_NONBLOCK, 0,
		      (struct stat *) 0, -1, -1, buf);
    if (qfile == 0) {
	if (errno != ENOENT)
	    msg_warn("open input file %s: %s", info->path, vstring_str(buf));
	vstring_free(buf);
	if (errno == EACCES)
	    msg_warn("if this file was created by Postfix < 1.1, then you may have to chmod a+r %s/%s",
		     var_queue_dir, info->path);
	return (errno == EACCES ? KEEP_MESSAGE_FILE : REMOVE_MESSAGE_FILE);
    }

    /*
     * Contact the cleanup service and read the queue ID that it has
     * allocated. In case of trouble, request that the cleanup service
     * bounces its copy of the message. because the original input file is
     * not readable by the bounce service.
     * 
     * If mail is re-injected with "postsuper -r", disable Milter applications.
     * If they were run before the mail was queued then there is no need to
     * run them again. Moreover, the queue file does not contain enough
     * information to reproduce the exact same SMTP events and Sendmail
     * macros that Milters received when the mail originally arrived in
     * Postfix.
     * 
     * The actual message copying code is in a separate routine, so that it is
     * easier to implement the many possible error exits without forgetting
     * to close files, or to release memory.
     */
    cleanup_flags =
	input_transp_cleanup(CLEANUP_FLAG_BOUNCE | CLEANUP_FLAG_MASK_EXTERNAL,
			     pickup_input_transp_mask);
    /* As documented in postsuper(1). */
    if (MAIL_IS_REQUEUED(info))
	cleanup_flags &= ~CLEANUP_FLAG_MILTER;
    else
	cleanup_flags |= smtputf8_autodetect(MAIL_SRC_MASK_SENDMAIL);

    cleanup = mail_connect_wait(MAIL_CLASS_PUBLIC, var_cleanup_service);
    if (attr_scan(cleanup, ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_QUEUEID, buf),
		  ATTR_TYPE_END) != 1
	|| attr_print(cleanup, ATTR_FLAG_NONE,
		      SEND_ATTR_INT(MAIL_ATTR_FLAGS, cleanup_flags),
		      ATTR_TYPE_END) != 0) {
	status = KEEP_MESSAGE_FILE;
    } else {
	info->id = mystrdup(vstring_str(buf));
	status = pickup_copy(qfile, cleanup, info, buf);
    }
    vstream_fclose(qfile);
    vstream_fclose(cleanup);
    vstring_free(buf);
    return (status);
}

/* pickup_init - init info structure */

static void pickup_init(PICKUP_INFO *info)
{
    info->id = 0;
    info->path = 0;
    info->sender = 0;
}

/* pickup_free - wipe info structure */

static void pickup_free(PICKUP_INFO *info)
{
#define SAFE_FREE(x) { if (x) myfree(x); }

    SAFE_FREE(info->id);
    SAFE_FREE(info->path);
    SAFE_FREE(info->sender);
}

/* pickup_service - service client */

static void pickup_service(char *unused_buf, ssize_t unused_len,
			           char *unused_service, char **argv)
{
    SCAN_DIR *scan;
    char   *queue_name;
    PICKUP_INFO info;
    const char *path;
    char   *id;
    int     file_count;

    /*
     * Sanity check. This service takes no command-line arguments.
     */
    if (argv[0])
	msg_fatal("unexpected command-line argument: %s", argv[0]);

    /*
     * Skip over things that we don't want to open, such as files that are
     * still being written, or garbage. Leave it up to the sysadmin to remove
     * garbage. Keep scanning the queue directory until we stop removing
     * files from it.
     * 
     * When we find a file, stroke the watchdog so that it will not bark while
     * some application is keeping us busy by injecting lots of mail into the
     * maildrop directory.
     */
    queue_name = MAIL_QUEUE_MAILDROP;		/* XXX should be a list */
    do {
	file_count = 0;
	scan = scan_dir_open(queue_name);
	while ((id = scan_dir_next(scan)) != 0) {
	    if (mail_open_ok(queue_name, id, &info.st, &path) == MAIL_OPEN_YES) {
		pickup_init(&info);
		info.path = mystrdup(path);
		watchdog_pat();
		if (pickup_file(&info) == REMOVE_MESSAGE_FILE) {
		    if (REMOVE(info.path))
			msg_warn("remove %s: %m", info.path);
		    else
			file_count++;
		}
		pickup_free(&info);
	    }
	}
	scan_dir_close(scan);
    } while (file_count);
}

/* post_jail_init - drop privileges */

static void post_jail_init(char *unused_name, char **unused_argv)
{

    /*
     * In case master.cf was not updated for unprivileged service.
     */
    if (getuid() != var_owner_uid)
	set_ugid(var_owner_uid, var_owner_gid);

    /*
     * Initialize the receive transparency options: do we want unknown
     * recipient checks, do we want address mapping.
     */
    pickup_input_transp_mask =
	input_transp_mask(VAR_INPUT_TRANSP, var_input_transp);
}

MAIL_VERSION_STAMP_DECLARE;

/* main - pass control to the multi-threaded server skeleton */

int     main(int argc, char **argv)
{
    static const CONFIG_STR_TABLE str_table[] = {
	VAR_FILTER_XPORT, DEF_FILTER_XPORT, &var_filter_xport, 0, 0,
	VAR_INPUT_TRANSP, DEF_INPUT_TRANSP, &var_input_transp, 0, 0,
	0,
    };

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    /*
     * Use the multi-threaded skeleton, because no-one else should be
     * monitoring our service socket while this process runs.
     * 
     * XXX The default watchdog timeout for trigger servers is 1000s, while the
     * cleanup server watchdog timeout is $daemon_timeout (i.e. several
     * hours). We override the default 1000s timeout to avoid problems with
     * slow mail submission. The real problem is of course that the
     * single-threaded pickup server is not a good solution for mail
     * submissions.
     */
    trigger_server_main(argc, argv, pickup_service,
			CA_MAIL_SERVER_STR_TABLE(str_table),
			CA_MAIL_SERVER_POST_INIT(post_jail_init),
			CA_MAIL_SERVER_SOLITARY,
			CA_MAIL_SERVER_WATCHDOG(&var_daemon_timeout),
			0);
}
