/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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
#ifndef SUDOERS_LOG_CLIENT_H
#define SUDOERS_LOG_CLIENT_H

#include <netinet/in.h>			/* for INET6?_ADDRSTRLEN */
#if defined(HAVE_OPENSSL)
# if defined(HAVE_WOLFSSL)
#  include <wolfssl/options.h>
# endif /* HAVE_WOLFSSL */
# include <openssl/ssl.h>
#endif /* HAVE_OPENSSL */

#include "log_server.pb-c.h"

#ifndef INET_ADDRSTRLEN
# define INET_ADDRSTRLEN 16
#endif
#ifndef INET6_ADDRSTRLEN
# define INET6_ADDRSTRLEN 46
#endif

#if PROTOBUF_C_VERSION_NUMBER < 1003000
# error protobuf-c version 1.30 or higher required
#endif

/* Default ports to listen on */
#define DEFAULT_PORT		"30343"
#define DEFAULT_PORT_TLS	"30344"

/* Maximum message size (2Mb) */
#define MESSAGE_SIZE_MAX	(2 * 1024 * 1024)

/* TODO - share with logsrvd/sendlog */
struct connection_buffer {
    TAILQ_ENTRY(connection_buffer) entries;
    uint8_t *data;
    unsigned int size;
    unsigned int len;
    unsigned int off;
};
TAILQ_HEAD(connection_buffer_list, connection_buffer);

enum client_state {
    ERROR,
    RECV_HELLO,
    SEND_RESTART,	/* TODO: currently unimplemented */
    SEND_ACCEPT,
    SEND_ALERT,
    SEND_REJECT,
    SEND_IO,
    SEND_EXIT,
    CLOSING,
    FINISHED
};

/* Remote connection closure, non-zero fields must come first. */
struct client_closure {
    int sock;
    bool read_instead_of_write;
    bool write_instead_of_read;
    bool temporary_write_event;
    bool disabled;
    bool log_io;
    char *server_name;
#if defined(HAVE_STRUCT_IN6_ADDR)
    char server_ip[INET6_ADDRSTRLEN];
#else
    char server_ip[INET_ADDRSTRLEN];
#endif
#if defined(HAVE_OPENSSL)
    SSL_CTX *ssl_ctx;
    SSL *ssl;
    bool ssl_initialized;
#endif /* HAVE_OPENSSL */
    bool subcommands;
    enum client_state state;
    enum client_state initial_state; /* XXX - bad name */
    struct connection_buffer_list write_bufs;
    struct connection_buffer_list free_bufs;
    struct connection_buffer read_buf;
    struct sudo_plugin_event *read_ev;
    struct sudo_plugin_event *write_ev;
    struct log_details *log_details;
    struct timespec start_time;
    struct timespec elapsed;
    struct timespec committed;
    char *iolog_id;
    const char *reason;
};

/* iolog_client.c */
struct client_closure *log_server_open(struct log_details *details, struct timespec *now, bool log_io, enum client_state initial_state, const char *reason);
bool log_server_close(struct client_closure *closure, int exit_status, int error);
bool fmt_client_message(struct client_closure *closure, ClientMessage *msg);
bool fmt_accept_message(struct client_closure *closure, struct eventlog *evlog);
bool fmt_reject_message(struct client_closure *closure, struct eventlog *evlog);
bool fmt_exit_message(struct client_closure *closure, int exit_status, int error);
bool fmt_io_buf(struct client_closure *closure, int type, const char *buf, unsigned int len, struct timespec *delay);
bool fmt_suspend(struct client_closure *closure, const char *signame, struct timespec *delay);
bool fmt_winsize(struct client_closure *closure, unsigned int lines, unsigned int cols, struct timespec *delay);
bool log_server_connect(struct client_closure *closure);
void client_closure_free(struct client_closure *closure);
bool read_server_hello(struct client_closure *closure);
extern struct client_closure *client_closure;

#endif /* SUDOERS_LOG_CLIENT_H */
