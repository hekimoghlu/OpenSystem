/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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

//
//  lib_fsck_msdos.h
//  fsck_msdos
//
//  Created by Kujan Lauz on 25/08/2022.
//

#ifndef lib_fsck_msdos_h
#define lib_fsck_msdos_h

#include <stdarg.h>
#include <sys/syslog.h>
#include <CoreFoundation/CoreFoundation.h>

typedef void* fsck_client_ctx_t;

/** Prints message */
typedef void (*fsck_msdos_print_funct_t)(fsck_client_ctx_t, int level, const char *fmt, va_list ap);

/** Asks the user a YES/NO question */
typedef int (*fsck_msdos_ask_func_t)(fsck_client_ctx_t, int def, const char *fmt, va_list ap);

/** Struct containing pointer functions to our print functions */
typedef struct {
	fsck_msdos_print_funct_t print;
	fsck_msdos_ask_func_t ask;
	fsck_client_ctx_t client_ctx;
} lib_fsck_ctx_t;

/** Sets up the print/ask functions, this method should be called before running checkfilesystem */
void fsck_set_context_properties(fsck_msdos_print_funct_t print,
								 fsck_msdos_ask_func_t ask,
								 fsck_client_ctx_t client);

void fsck_set_alwaysyes(bool alwaysyes);
bool fsck_alwaysyes(void);

void fsck_set_alwaysno(bool alwaysno);
bool fsck_alwaysno(void);

void fsck_set_preen(bool preen);
bool fsck_preen(void);

void fsck_set_quick(bool quick);
bool fsck_quick(void);

void fsck_set_quiet(bool quiet);
bool fsck_quiet(void);

void fsck_set_rdonly(bool rdonly);
bool fsck_rdonly(void);

void fsck_set_maxmem(size_t maxmem);
size_t fsck_maxmem(void);

void fsck_set_dev(const char* dev);
const char* fsck_dev(void);

void fsck_set_fd(int fd);
int fsck_fd(void);

void fsck_print(lib_fsck_ctx_t, int level, const char *fmt, ...) __printflike(3, 4);
int fsck_ask(lib_fsck_ctx_t, int def, const char *fmt, ...) __printflike(3, 4);

/** Instance containing the function pointers for running fsck_msdos */
extern lib_fsck_ctx_t fsck_ctx;

#endif /* lib_fsck_msdos_h */
