/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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
#ifndef BC_READ_H
#define BC_READ_H

#include <stdlib.h>

#include <status.h>
#include <vector.h>

/**
 * Returns true if @a c is a non-ASCII (invalid) char.
 * @param c  The character to test.
 * @return   True if @a c is an invalid char.
 */
#define BC_READ_BIN_CHAR(c) (!(c))

/**
 * Reads a line from stdin after printing prompt, if desired.
 * @param vec     The vector to put the stdin data into.
 * @param prompt  The prompt to print, if desired.
 */
BcStatus
bc_read_line(BcVec* vec, const char* prompt);

/**
 * Read a file and return a buffer with the data. The buffer must be freed by
 * the caller.
 * @param path  The path to the file to read.
 */
char*
bc_read_file(const char* path);

/**
 * Helper function for reading characters from stdin. This takes care of a bunch
 * of complex error handling. Thus, it returns a status instead of throwing an
 * error, except for fatal errors.
 * @param vec     The vec to put the stdin into.
 * @param prompt  The prompt to print, if desired.
 */
BcStatus
bc_read_chars(BcVec* vec, const char* prompt);

/**
 * Read a line from buf into vec.
 * @param vec      The vector to read data into.
 * @param buf      The buffer to read from.
 * @param buf_len  The length of the buffer.
 */
bool
bc_read_buf(BcVec* vec, char* buf, size_t* buf_len);

#endif // BC_READ_H
