/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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

#ifndef HEADER_CURL_SERVER_UTIL_H
#define HEADER_CURL_SERVER_UTIL_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 * SPDX-License-Identifier: curl
 *
 ***************************************************************************/
#include "server_setup.h"

char *data_to_hex(char *data, size_t len);
void logmsg(const char *msg, ...) CURL_PRINTF(1, 2);
long timediff(struct timeval newer, struct timeval older);

#define TEST_DATA_PATH "%s/data/test%ld"
#define ALTTEST_DATA_PATH "%s/test%ld"
#define SERVERLOGS_LOCKDIR "lock"  /* within logdir */

/* global variable, where to find the 'data' dir */
extern const char *path;

/* global variable, log file name */
extern const char *serverlogfile;

extern const char *cmdfile;

#ifdef _WIN32
#include <process.h>
#include <fcntl.h>

#define sleep(sec) Sleep ((sec)*1000)

#undef perror
#define perror(m) win32_perror(m)
void win32_perror(const char *msg);

void win32_init(void);
void win32_cleanup(void);
const char *sstrerror(int err);
#else   /* _WIN32 */

#define sstrerror(e) strerror(e)
#endif  /* _WIN32 */

/* fopens the test case file */
FILE *test2fopen(long testno, const char *logdir);

int wait_ms(int timeout_ms);
curl_off_t our_getpid(void);
int write_pidfile(const char *filename);
int write_portfile(const char *filename, int port);
void set_advisor_read_lock(const char *filename);
void clear_advisor_read_lock(const char *filename);

/* global variable which if set indicates that the program should finish */
extern volatile int got_exit_signal;

/* global variable which if set indicates the first signal handled */
extern volatile int exit_signal;

#ifdef _WIN32
/* global event which if set indicates that the program should finish */
extern HANDLE exit_event;
#endif

void install_signal_handlers(bool keep_sigalrm);
void restore_signal_handlers(bool keep_sigalrm);

#ifdef USE_UNIX_SOCKETS

#include <curl/curl.h> /* for curl_socket_t */

#ifdef HAVE_SYS_UN_H
#include <sys/un.h> /* for sockaddr_un */
#endif /* HAVE_SYS_UN_H */

int bind_unix_socket(curl_socket_t sock, const char *unix_socket,
        struct sockaddr_un *sau);
#endif  /* USE_UNIX_SOCKETS */

#endif  /* HEADER_CURL_SERVER_UTIL_H */
