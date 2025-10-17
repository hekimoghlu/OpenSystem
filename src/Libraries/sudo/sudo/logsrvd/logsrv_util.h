/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
#ifndef SUDO_LOGSRV_UTIL_H
#define SUDO_LOGSRV_UTIL_H

#include <netinet/in.h>		/* for INET_ADDRSTRLEN and INET6_ADDRSTRLEN */

#ifndef INET_ADDRSTRLEN
# define INET_ADDRSTRLEN 16
#endif
#ifndef INET6_ADDRSTRLEN
# define INET6_ADDRSTRLEN 46
#endif

/* Default ports to listen on */
#define DEFAULT_PORT		"30343"
#define DEFAULT_PORT_TLS	"30344"

/* Maximum message size (2Mb) */
#define MESSAGE_SIZE_MAX	(2 * 1024 * 1024)

struct peer_info {
    const char *name;
#if defined(HAVE_STRUCT_IN6_ADDR)
    char ipaddr[INET6_ADDRSTRLEN];
#else
    char ipaddr[INET_ADDRSTRLEN];
#endif
};

struct connection_buffer {
    TAILQ_ENTRY(connection_buffer) entries;
    uint8_t *data;
    unsigned int size;
    unsigned int len;
    unsigned int off;
};
TAILQ_HEAD(connection_buffer_list, connection_buffer);

/* logsrv_util.c */
struct iolog_file;
bool expand_buf(struct connection_buffer *buf, unsigned int needed);
bool iolog_open_all(int dfd, const char *iolog_dir, struct iolog_file *iolog_files, const char *mode);
bool iolog_seekto(int iolog_dir_fd, const char *iolog_path, struct iolog_file *iolog_files, struct timespec *elapsed_time, const struct timespec *target);


#endif /* SUDO_LOGSRV_UTIL_H */
