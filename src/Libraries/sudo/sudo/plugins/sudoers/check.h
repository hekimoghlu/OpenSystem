/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
#ifndef SUDOERS_CHECK_H
#define SUDOERS_CHECK_H

/* Status codes for timestamp_status() */
#define TS_CURRENT		0
#define TS_OLD			1
#define TS_MISSING		2
#define TS_ERROR		3
#define TS_FATAL		4

/*
 * Time stamps are now stored in a single file which contains multiple
 * records.  Each record starts with a 16-bit version number and a 16-bit
 * record size.  Multiple record types can coexist in the same file.
 */
#define	TS_VERSION		2

/* Time stamp entry types */
#define TS_GLOBAL		0x01	/* not restricted by tty or ppid */
#define TS_TTY			0x02	/* restricted by tty */
#define TS_PPID			0x03	/* restricted by ppid */
#define TS_LOCKEXCL		0x04	/* special lock record */

/* Time stamp flags */
#define TS_DISABLED		0x01	/* entry disabled */
#define TS_ANYUID		0x02	/* ignore uid, only valid in the key */

struct timestamp_entry_v1 {
    unsigned short version;	/* version number */
    unsigned short size;	/* entry size */
    unsigned short type;	/* TS_GLOBAL, TS_TTY, TS_PPID */
    unsigned short flags;	/* TS_DISABLED, TS_ANYUID */
    uid_t auth_uid;		/* uid to authenticate as */
    pid_t sid;			/* session ID associated with tty/ppid */
    struct timespec ts;		/* time stamp (CLOCK_MONOTONIC) */
    union {
	dev_t ttydev;		/* tty device number */
	pid_t ppid;		/* parent pid */
    } u;
};

struct timestamp_entry {
    unsigned short version;	/* version number */
    unsigned short size;	/* entry size */
    unsigned short type;	/* TS_GLOBAL, TS_TTY, TS_PPID */
    unsigned short flags;	/* TS_DISABLED, TS_ANYUID */
    uid_t auth_uid;		/* uid to authenticate as */
    pid_t sid;			/* session ID associated with tty/ppid */
    struct timespec start_time;	/* session/ppid start time */
    struct timespec ts;		/* time stamp (CLOCK_MONOTONIC) */
    union {
	dev_t ttydev;		/* tty device number */
	pid_t ppid;		/* parent pid */
    } u;
};

void *timestamp_open(const char *user, pid_t sid);
void  timestamp_close(void *vcookie);
bool  timestamp_lock(void *vcookie, struct passwd *pw);
bool  timestamp_update(void *vcookie, struct passwd *pw);
int   timestamp_status(void *vcookie, struct passwd *pw);
int   get_starttime(pid_t pid, struct timespec *starttime);
bool  already_lectured(void);
int   set_lectured(void);
void display_lecture(struct sudo_conv_callback *callback);
int   create_admin_success_flag(void);

#endif /* SUDOERS_CHECK_H */
