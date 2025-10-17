/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 26, 2025.
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
#ifndef SUDO_EVENTLOG_H
#define SUDO_EVENTLOG_H

#include <sys/types.h>	/* for gid_t, uid_t */
#include <time.h>	/* for struct timespec */
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif /* HAVE_STDBOOL_H */

/* Supported event types. */
enum event_type {
    EVLOG_ACCEPT,
    EVLOG_REJECT,
    EVLOG_EXIT,
    EVLOG_ALERT
};

/* Supported eventlog types (bitmask). */
#define EVLOG_NONE	0x00
#define EVLOG_SYSLOG	0x01
#define EVLOG_FILE	0x02

/* Supported eventlog formats. */
enum eventlog_format {
    EVLOG_SUDO,
    EVLOG_JSON
};

/* Eventlog flag values. */
#define EVLOG_RAW	0x01	/* only include message and errstr */
#define EVLOG_MAIL	0x02	/* mail the log message too */
#define EVLOG_MAIL_ONLY	0x04	/* only mail the message, no other logging */
#define EVLOG_CWD	0x08	/* log cwd if no runcwd and use CWD, not PWD */

/*
 * Maximum number of characters to log per entry.  The syslogger
 * will log this much, after that, it truncates the log line.
 * We need this here to make sure that we continue with another
 * syslog(3) call if the internal buffer is more than 1023 characters.
 */
#ifndef MAXSYSLOGLEN
# define MAXSYSLOGLEN		960
#endif

/*
 * Indentation level for file-based logs when word wrap is enabled.
 */
#define EVENTLOG_INDENT	"    "

/*
 * Event log config, used with eventlog_getconf()
 */
struct eventlog_config {
    int type;
    enum eventlog_format format;
    int syslog_acceptpri;
    int syslog_rejectpri;
    int syslog_alertpri;
    int syslog_maxlen;
    int file_maxlen;
    uid_t mailuid;
    bool omit_hostname;
    const char *logpath;
    const char *time_fmt;
    const char *mailerpath;
    const char *mailerflags;
    const char *mailfrom;
    const char *mailto;
    const char *mailsub;
    FILE *(*open_log)(int type, const char *);
    void (*close_log)(int type, FILE *);
};

/*
 * Info present in the eventlog file, regardless of format.
 */
struct eventlog {
    char *iolog_path;
    const char *iolog_file;	/* substring of iolog_path, do not free */
    char *command;
    char *cwd;
    char *runchroot;
    char *runcwd;
    char *rungroup;
    char *runuser;
    char *peeraddr;
    char *signal_name;
    char *submithost;
    char *submituser;
    char *submitgroup;
    char *ttyname;
    char **argv;
    char **env_add;
    char **envp;
    struct timespec submit_time;
    struct timespec iolog_offset;
    struct timespec run_time;
    int exit_value;
    int lines;
    int columns;
    uid_t runuid;
    gid_t rungid;
    bool dumped_core;
    char sessid[7];
    char uuid_str[37];
};

/* Callback from eventlog code to write log info */
struct json_container;
struct sudo_lbuf;
typedef bool (*eventlog_json_callback_t)(struct json_container *, void *);

bool eventlog_accept(const struct eventlog *evlog, int flags, eventlog_json_callback_t info_cb, void *info);
bool eventlog_exit(const struct eventlog *evlog, int flags);
bool eventlog_alert(const struct eventlog *evlog, int flags, struct timespec *alert_time, const char *reason, const char *errstr);
bool eventlog_mail(const struct eventlog *evlog, int flags, struct timespec *event_time, const char *reason, const char *errstr, char * const extra[]);
bool eventlog_reject(const struct eventlog *evlog, int flags, const char *reason, eventlog_json_callback_t info_cb, void *info);
bool eventlog_store_json(struct json_container *jsonc, const struct eventlog *evlog);
bool eventlog_store_sudo(int event_type, const struct eventlog *evlog, struct sudo_lbuf *lbuf);
size_t eventlog_writeln(FILE *fp, char *line, size_t len, size_t maxlen);
void eventlog_free(struct eventlog *evlog);
void eventlog_set_type(int type);
void eventlog_set_format(enum eventlog_format format);
void eventlog_set_syslog_acceptpri(int pri);
void eventlog_set_syslog_rejectpri(int pri);
void eventlog_set_syslog_alertpri(int pri);
void eventlog_set_syslog_maxlen(int len);
void eventlog_set_file_maxlen(int len);
void eventlog_set_mailuid(uid_t uid);
void eventlog_set_omit_hostname(bool omit_hostname);
void eventlog_set_logpath(const char *path);
void eventlog_set_time_fmt(const char *fmt);
void eventlog_set_mailerpath(const char *path);
void eventlog_set_mailerflags(const char *mflags);
void eventlog_set_mailfrom(const char *from_addr);
void eventlog_set_mailto(const char *to_addr);
void eventlog_set_mailsub(const char *subject);
void eventlog_set_open_log(FILE *(*fn)(int type, const char *));
void eventlog_set_close_log(void (*fn)(int type, FILE *));
const struct eventlog_config *eventlog_getconf(void);

#endif /* SUDO_EVENTLOG_H */
