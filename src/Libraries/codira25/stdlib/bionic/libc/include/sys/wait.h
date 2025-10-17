/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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

#include <sys/cdefs.h>

#include <bits/wait.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <linux/wait.h>
#include <signal.h>

__BEGIN_DECLS

pid_t wait(int* _Nullable __status);
pid_t waitpid(pid_t __pid, int* _Nullable __status, int __options);
pid_t wait4(pid_t __pid, int* _Nullable __status, int __options, struct rusage* _Nullable __rusage);

/* Posix states that idtype_t should be an enumeration type, but
 * the kernel headers define P_ALL, P_PID and P_PGID as constant macros
 * instead.
 */
typedef int idtype_t;

int waitid(idtype_t __type, id_t __id, siginfo_t* _Nullable __info, int __options);

__END_DECLS
