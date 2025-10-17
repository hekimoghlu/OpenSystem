/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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

__BEGIN_DECLS

/**
 * [getopt(3)](https://man7.org/linux/man-pages/man3/getopt.3.html) parses command-line options.
 *
 * Returns the next option character on success, returns -1 if all options have been parsed, and
 * returns `'?'` on error.
 */
int getopt(int __argc, char* const _Nonnull __argv[_Nullable], const char* _Nonnull __options);

/**
 * Points to the text of the corresponding value for options that take an argument.
 */
extern char* _Nullable optarg;

/**
 * The index of the next element to be processed.
 * On Android, callers should set `optreset = 1` rather than trying to reset `optind` to
 * scan a new argument vector.
 */
extern int optind;

/**
 * Determines whether getopt() outputs error messages.
 * Callers should set this to `0` to disable error messages.
 * Defaults to non-zero.
 */
extern int opterr;

/**
 * The last unrecognized option character, valid when getopt() returns `'?'`.
 */
extern int optopt;

__END_DECLS
