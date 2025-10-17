/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
#ifndef SUDO_SENDLOG_H
#define SUDO_SENDLOG_H

#include "log_server.pb-c.h"
#if PROTOBUF_C_VERSION_NUMBER < 1003000
# error protobuf-c version 1.30 or higher required
#endif

#include <config.h>

#if defined(HAVE_OPENSSL)
# if defined(HAVE_WOLFSSL)
#  include <wolfssl/options.h>
# endif
# include <openssl/ssl.h>
# include <openssl/err.h>
#endif

#include "logsrv_util.h"
#include "tls_common.h"

enum client_state {
    ERROR,
    RECV_HELLO,
    SEND_RESTART,
    SEND_ACCEPT,
    SEND_REJECT,
    SEND_IO,
    SEND_EXIT,
    CLOSING,
    FINISHED
};

struct client_closure {
    TAILQ_ENTRY(client_closure) entries;
    int sock;
    bool accept_only;
    bool read_instead_of_write;
    bool write_instead_of_read;
    bool temporary_write_event;
    struct timespec restart;
    struct timespec stop_after;
    struct timespec elapsed;
    struct timespec committed;
    struct timing_closure timing;
    struct sudo_event_base *evbase;
    struct connection_buffer read_buf;
    struct connection_buffer_list write_bufs;
    struct connection_buffer_list free_bufs;
#if defined(HAVE_OPENSSL)
    struct tls_client_closure tls_client;
#endif
    struct sudo_event *read_ev;
    struct sudo_event *write_ev;
    struct eventlog *evlog;
    struct iolog_file iolog_files[IOFD_MAX];
    const char *iolog_id;
    char *reject_reason;
    char *buf; /* XXX */
    size_t bufsize; /* XXX */
    enum client_state state;
};

#endif /* SUDO_SENDLOG_H */
