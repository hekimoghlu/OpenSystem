/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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
 * This header is used to define signal constants and names;
 * it might be included several times.
 */

#ifndef __BIONIC_SIGDEF
#error __BIONIC_SIGDEF not defined
#endif

__BIONIC_SIGDEF(SIGHUP,    "Hangup")
__BIONIC_SIGDEF(SIGINT,    "Interrupt")
__BIONIC_SIGDEF(SIGQUIT,   "Quit")
__BIONIC_SIGDEF(SIGILL,    "Illegal instruction")
__BIONIC_SIGDEF(SIGTRAP,   "Trap")
__BIONIC_SIGDEF(SIGABRT,   "Aborted")
__BIONIC_SIGDEF(SIGFPE,    "Floating point exception")
__BIONIC_SIGDEF(SIGKILL,   "Killed")
__BIONIC_SIGDEF(SIGBUS,    "Bus error")
__BIONIC_SIGDEF(SIGSEGV,   "Segmentation fault")
__BIONIC_SIGDEF(SIGPIPE,   "Broken pipe")
__BIONIC_SIGDEF(SIGALRM,   "Alarm clock")
__BIONIC_SIGDEF(SIGTERM,   "Terminated")
__BIONIC_SIGDEF(SIGUSR1,   "User signal 1")
__BIONIC_SIGDEF(SIGUSR2,   "User signal 2")
__BIONIC_SIGDEF(SIGCHLD,   "Child exited")
__BIONIC_SIGDEF(SIGPWR,    "Power failure")
__BIONIC_SIGDEF(SIGWINCH,  "Window size changed")
__BIONIC_SIGDEF(SIGURG,    "Urgent I/O condition")
__BIONIC_SIGDEF(SIGIO,     "I/O possible")
__BIONIC_SIGDEF(SIGSTOP,   "Stopped (signal)")
__BIONIC_SIGDEF(SIGTSTP,   "Stopped")
__BIONIC_SIGDEF(SIGCONT,   "Continue")
__BIONIC_SIGDEF(SIGTTIN,   "Stopped (tty input)")
__BIONIC_SIGDEF(SIGTTOU,   "Stopped (tty output)")
__BIONIC_SIGDEF(SIGVTALRM, "Virtual timer expired")
__BIONIC_SIGDEF(SIGPROF,   "Profiling timer expired")
__BIONIC_SIGDEF(SIGXCPU,   "CPU time limit exceeded")
__BIONIC_SIGDEF(SIGXFSZ,   "File size limit exceeded")
__BIONIC_SIGDEF(SIGSTKFLT, "Stack fault")
__BIONIC_SIGDEF(SIGSYS,    "Bad system call")

#undef __BIONIC_SIGDEF
