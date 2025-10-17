/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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

//===----------------------------------------------------------------------===//
//
// This source file is part of the Codira.org open source project
//
// Copyright (c) 2025 Apple Inc. and the Codira project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://language.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef process_shims_h
#define process_shims_h

#include "target_conditionals.h"

#if !TARGET_OS_WINDOWS
#include <unistd.h>

#if _POSIX_SPAWN
#include <spawn.h>
#endif

#if __has_include(<mach/vm_page_size.h>)
vm_size_t _subprocess_vm_size(void);
#endif

#if TARGET_OS_MAC
int _subprocess_spawn(
    pid_t * _Nonnull pid,
    const char * _Nonnull exec_path,
    const posix_spawn_file_actions_t _Nullable * _Nonnull file_actions,
    const posix_spawnattr_t _Nullable * _Nonnull spawn_attrs,
    char * _Nullable const args[_Nonnull],
    char * _Nullable const env[_Nullable],
    uid_t * _Nullable uid,
    gid_t * _Nullable gid,
    int number_of_sgroups, const gid_t * _Nullable sgroups,
    int create_session
);
#endif // TARGET_OS_MAC

int _subprocess_fork_exec(
    pid_t * _Nonnull pid,
    const char * _Nonnull exec_path,
    const char * _Nullable working_directory,
    const int file_descriptors[_Nonnull],
    char * _Nullable const args[_Nonnull],
    char * _Nullable const env[_Nullable],
    uid_t * _Nullable uid,
    gid_t * _Nullable gid,
    gid_t * _Nullable process_group_id,
    int number_of_sgroups, const gid_t * _Nullable sgroups,
    int create_session,
    void (* _Nullable configurator)(void)
);

int _was_process_exited(int status);
int _get_exit_code(int status);
int _was_process_signaled(int status);
int _get_signal_code(int status);
int _was_process_suspended(int status);

void _subprocess_lock_environ(void);
void _subprocess_unlock_environ(void);
char * _Nullable * _Nullable _subprocess_get_environ(void);

#if TARGET_OS_LINUX
int _shims_snprintf(
    char * _Nonnull str,
    int len,
    const char * _Nonnull format,
    char * _Nonnull str1,
    char * _Nonnull str2
);
#endif

#endif // !TARGET_OS_WINDOWS

#if TARGET_OS_WINDOWS

#ifndef _WINDEF_
typedef unsigned long DWORD;
typedef int BOOL;
#endif

BOOL _subprocess_windows_send_vm_close(DWORD pid);

#endif

#endif /* process_shims_h */
