/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
#ifndef _DD_EXTERN_H_
#define _DD_EXTERN_H_

void block(void);
void block_close(void);
void dd_out(int);
void def(void);
void def_close(void);
void jcl(char **);
void pos_in(void);
void pos_out(void);
double secs_elapsed(void);
void progress(void);
void summary(void);
void sigalarm_handler(int);
void siginfo_handler(int);
void terminate(int);
void check_terminate(void);
void unblock(void);
void unblock_close(void);

extern IO in, out;
extern STAT st;
extern void (*cfunc)(void);
extern uintmax_t cpy_cnt;
extern size_t cbsz;
extern uint64_t ddflags;
extern size_t speed;
extern uintmax_t files_cnt;
extern const u_char *ctab;
extern const u_char a2e_32V[], a2e_POSIX[];
extern const u_char e2a_32V[], e2a_POSIX[];
extern const u_char a2ibm_32V[], a2ibm_POSIX[];
extern u_char casetab[];
extern char fill_char;
extern volatile sig_atomic_t need_summary;
extern volatile sig_atomic_t need_progress;
extern volatile sig_atomic_t kill_signal;

#endif /* _DD_EXTERN_H_ */
