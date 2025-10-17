/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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
#ifndef SUDO_EXEC_INTERCEPT_H
#define SUDO_EXEC_INTERCEPT_H

enum intercept_state {
    INVALID_STATE,
    RECV_HELLO_INITIAL,
    RECV_HELLO,
    RECV_SECRET,
    RECV_POLICY_CHECK,
    RECV_CONNECTION,
    POLICY_ACCEPT,
    POLICY_REJECT,
    POLICY_TEST,
    POLICY_ERROR
};

/* Closure for intercept_cb() */
struct intercept_closure {
    union sudo_token_un token;
    struct command_details *details;
    struct sudo_event ev;
    const char *errstr;
    char *command;		/* dynamically allocated */
    char **run_argv;		/* owned by plugin */
    char **run_envp;		/* dynamically allocated */
    uint8_t *buf;		/* dynamically allocated */
    uint32_t len;
    uint32_t off;
    int listen_sock;
    enum intercept_state state;
    int initial_command;
};

void intercept_closure_reset(struct intercept_closure *closure);
bool intercept_check_policy(const char *command, int argc, char **argv, int envc, char **envp, const char *runcwd, int *oldcwd, void *closure);

#endif /* SUDO_EXEC_INTERCEPT_H */
