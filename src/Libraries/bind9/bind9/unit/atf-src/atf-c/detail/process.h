/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#if !defined(ATF_C_PROCESS_H)
#define ATF_C_PROCESS_H

#include <sys/types.h>

#include <stdbool.h>

#include <atf-c/error_fwd.h>

#include "fs.h"
#include "list.h"

/* ---------------------------------------------------------------------
 * The "atf_process_stream" type.
 * --------------------------------------------------------------------- */

struct atf_process_stream {
    int m_type;

    /* Valid if m_type == connect. */
    int m_src_fd;
    int m_tgt_fd;

    /* Valid if m_type == redirect_fd. */
    int m_fd;

    /* Valid if m_type == redirect_path. */
    const atf_fs_path_t *m_path;
};
typedef struct atf_process_stream atf_process_stream_t;

extern const int atf_process_stream_type_capture;
extern const int atf_process_stream_type_connect;
extern const int atf_process_stream_type_inherit;
extern const int atf_process_stream_type_redirect_fd;
extern const int atf_process_stream_type_redirect_path;

atf_error_t atf_process_stream_init_capture(atf_process_stream_t *);
atf_error_t atf_process_stream_init_connect(atf_process_stream_t *,
                                            const int, const int);
atf_error_t atf_process_stream_init_inherit(atf_process_stream_t *);
atf_error_t atf_process_stream_init_redirect_fd(atf_process_stream_t *,
                                                const int fd);
atf_error_t atf_process_stream_init_redirect_path(atf_process_stream_t *,
                                                  const atf_fs_path_t *);
void atf_process_stream_fini(atf_process_stream_t *);

int atf_process_stream_type(const atf_process_stream_t *);

/* ---------------------------------------------------------------------
 * The "atf_process_status" type.
 * --------------------------------------------------------------------- */

struct atf_process_status {
    int m_status;
};
typedef struct atf_process_status atf_process_status_t;

void atf_process_status_fini(atf_process_status_t *);

bool atf_process_status_exited(const atf_process_status_t *);
int atf_process_status_exitstatus(const atf_process_status_t *);
bool atf_process_status_signaled(const atf_process_status_t *);
int atf_process_status_termsig(const atf_process_status_t *);
bool atf_process_status_coredump(const atf_process_status_t *);

/* ---------------------------------------------------------------------
 * The "atf_process_child" type.
 * --------------------------------------------------------------------- */

struct atf_process_child {
    pid_t m_pid;

    int m_stdout;
    int m_stderr;
};
typedef struct atf_process_child atf_process_child_t;

atf_error_t atf_process_child_wait(atf_process_child_t *,
                                   atf_process_status_t *);
pid_t atf_process_child_pid(const atf_process_child_t *);
int atf_process_child_stdout(atf_process_child_t *);
int atf_process_child_stderr(atf_process_child_t *);

/* ---------------------------------------------------------------------
 * Free functions.
 * --------------------------------------------------------------------- */

atf_error_t atf_process_fork(atf_process_child_t *,
                             void (*)(void *),
                             const atf_process_stream_t *,
                             const atf_process_stream_t *,
                             void *);
atf_error_t atf_process_exec_array(atf_process_status_t *,
                                   const atf_fs_path_t *,
                                   const char *const *,
                                   const atf_process_stream_t *,
                                   const atf_process_stream_t *,
                                   void (*)(void));
atf_error_t atf_process_exec_list(atf_process_status_t *,
                                  const atf_fs_path_t *,
                                  const atf_list_t *,
                                  const atf_process_stream_t *,
                                  const atf_process_stream_t *,
                                  void (*)(void));

#endif /* !defined(ATF_C_PROCESS_H) */
