/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
#if !defined(ATF_C_UTILS_H)
#define ATF_C_UTILS_H

#include <stdbool.h>
#include <unistd.h>

#include <atf-c/defs.h>

void atf_utils_cat_file(const char *, const char *);
bool atf_utils_compare_file(const char *, const char *);
void atf_utils_copy_file(const char *, const char *);
void atf_utils_create_file(const char *, const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(2, 3);
bool atf_utils_file_exists(const char *);
pid_t atf_utils_fork(void);
void atf_utils_free_charpp(char **);
bool atf_utils_grep_file(const char *, const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(1, 3);
bool atf_utils_grep_string(const char *, const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(1, 3);
char *atf_utils_readline(int);
void atf_utils_redirect(const int, const char *);
void atf_utils_wait(const pid_t, const int, const char *, const char *);

#endif /* ATF_C_UTILS_H */
