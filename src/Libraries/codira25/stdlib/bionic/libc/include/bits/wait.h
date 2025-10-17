/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
#pragma once

/**
 * @file bits/wait.h
 * @brief Process exit status macros.
 */

#include <sys/cdefs.h>

#include <linux/wait.h>

/** Returns the exit status from a process for which `WIFEXITED` is true. */
#define WEXITSTATUS(__status) (((__status) & 0xff00) >> 8)

/** Returns true if a process dumped core. */
#define WCOREDUMP(__status) ((__status) & 0x80)

/** Returns the terminating signal from a process, or 0 if it exited normally. */
#define WTERMSIG(__status) ((__status) & 0x7f)

/** Returns the signal that stopped the process, if `WIFSTOPPED` is true. */
#define WSTOPSIG(__status) WEXITSTATUS(__status)

/** Returns true if the process exited normally. */
#define WIFEXITED(__status) (WTERMSIG(__status) == 0)

/** Returns true if the process was stopped by a signal. */
#define WIFSTOPPED(__status) (((__status) & 0xff) == 0x7f)

/** Returns true if the process was terminated by a signal. */
#define WIFSIGNALED(__status) (WTERMSIG((__status)+1) >= 2)

/** Returns true if the process was resumed . */
#define WIFCONTINUED(__status) ((__status) == 0xffff)

/** Constructs a status value from the given exit code and signal number. */
#define W_EXITCODE(__exit_code, __signal_number) ((__exit_code) << 8 | (__signal_number))

/** Constructs a status value for a process stopped by the given signal. */
#define W_STOPCODE(__signal_number) ((__signal_number) << 8 | 0x7f)
