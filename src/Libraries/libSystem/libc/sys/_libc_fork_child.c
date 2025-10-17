/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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
/*
 * _libc_fork_child() is called from Libsystem's libSystem_atfork_child()
 */
#include <TargetConditionals.h>
#if __has_include(<CrashReporterClient.h>)
#include <CrashReporterClient.h>
#else
#define CRSetCrashLogMessage(...)
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstrict-prototypes"

extern void _arc4_fork_child();
extern void _init_clock_port(void);
extern void __environ_lock_fork_child();

void _libc_fork_child(void); // todo: private_extern?
void
_libc_fork_child(void)
{
	CRSetCrashLogMessage("crashed on child side of fork pre-exec");

	_arc4_fork_child();
	_init_clock_port();
	__environ_lock_fork_child();
}
#pragma clang diagnostic pop
