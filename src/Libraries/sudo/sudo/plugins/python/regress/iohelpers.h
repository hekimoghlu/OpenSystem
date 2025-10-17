/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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
#ifndef PYTHON_IO_HELPERS
#define PYTHON_IO_HELPERS

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif /* HAVE_STDBOOL_H */
#include <string.h>
#include <stdarg.h>
#include <signal.h>
#include <pwd.h>

#include "sudo_compat.h"

#define MAX_OUTPUT (2 << 16)

int rmdir_recursive(const char *path);

int fwriteall(const char *file_path, const char *string);
int freadall(const char *file_path, char *output, size_t max_len);

// allocates new string with the content of 'string' but 'old' replaced to 'new'
// The allocated array will be dest_length size and null terminated correctly.
char *str_replaced(const char *string, size_t dest_length, const char *old, const char *new);

// same, but "string" must be able to store 'max_length' number of characters including the null terminator
void str_replace_in_place(char *string, size_t max_length, const char *old, const char *new);

int vsnprintf_append(char *output, size_t max_output_len, const char *fmt, va_list args);
int snprintf_append(char *output, size_t max_output_len, const char *fmt, ...);

int str_array_count(char **str_array);
void str_array_snprint(char *out_str, size_t max_len, char **str_array, int array_len);

#endif
