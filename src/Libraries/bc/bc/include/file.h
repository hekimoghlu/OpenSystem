/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#ifndef BC_FILE_H
#define BC_FILE_H

#include <stdarg.h>

#include <history.h>
#include <vector.h>

#define BC_FILE_ULL_LENGTH (21)

#if BC_ENABLE_LINE_LIB

#include <stdio.h>

/// The file struct.
typedef struct BcFile
{
	// The file. This is here simply to make the line lib code as compatible
	// with the existing code as possible.
	FILE* f;

	// True if errors should be fatal, false otherwise.
	bool errors_fatal;

} BcFile;

#else // BC_ENABLE_LINE_LIB

/// The file struct.
typedef struct BcFile
{
	// The actual file descriptor.
	int fd;

	// True if errors should be fatal, false otherwise.
	bool errors_fatal;

	// The buffer for the file.
	char* buf;

	// The length (number of actual chars) in the buffer.
	size_t len;

	// The capacity (max number of chars) of the buffer.
	size_t cap;

} BcFile;

#endif // BC_ENABLE_LINE_LIB

#if BC_ENABLE_HISTORY && !BC_ENABLE_LINE_LIB

/// Types of flushing. These are important because of history and printing
/// strings without newlines, something that users could use as their own
/// prompts.
typedef enum BcFlushType
{
	/// Do not clear the stored partial line, but don't add to it.
	BC_FLUSH_NO_EXTRAS_NO_CLEAR,

	/// Do not clear the stored partial line and add to it.
	BC_FLUSH_SAVE_EXTRAS_NO_CLEAR,

	/// Clear the stored partial line and do not save the new stuff either.
	BC_FLUSH_NO_EXTRAS_CLEAR,

	/// Clear the stored partial line, but save the new stuff.
	BC_FLUSH_SAVE_EXTRAS_CLEAR,

} BcFlushType;

// These are here to satisfy a clang warning about recursive macros.

#define bc_file_putchar(f, t, c) bc_file_putchar_impl(f, t, c)
#define bc_file_flushErr(f, t) bc_file_flushErr_impl(f, t)
#define bc_file_flush(f, t) bc_file_flush_impl(f, t)
#define bc_file_write(f, t, b, n) bc_file_write_impl(f, t, b, n)
#define bc_file_puts(f, t, s) bc_file_puts_impl(f, t, s)

#else // BC_ENABLE_HISTORY && !BC_ENABLE_LINE_LIB

// These make sure that the BcFlushType parameter disappears if history is not
// used, editline is used, or readline is used.

#define bc_file_putchar(f, t, c) bc_file_putchar_impl(f, c)
#define bc_file_flushErr(f, t) bc_file_flushErr_impl(f)
#define bc_file_flush(f, t) bc_file_flush_impl(f)
#define bc_file_write(f, t, b, n) bc_file_write_impl(f, b, n)
#define bc_file_puts(f, t, s) bc_file_puts_impl(f, s)

#endif // BC_ENABLE_HISTORY && !BC_ENABLE_LINE_LIB

#if BC_ENABLE_LINE_LIB

/**
 * Initialize a file.
 * @param f             The file to initialize.
 * @param file          The stdio file.
 * @param errors_fatal  True if errors should be fatal, false otherwise.
 */
void
bc_file_init(BcFile* f, FILE* file, bool errors_fatal);

#else // BC_ENABLE_LINE_LIB

/**
 * Initialize a file.
 * @param f             The file to initialize.
 * @param fd            The file descriptor.
 * @param buf           The buffer for the file.
 * @param cap           The capacity of the buffer.
 * @param errors_fatal  True if errors should be fatal, false otherwise.
 */
void
bc_file_init(BcFile* f, int fd, char* buf, size_t cap, bool errors_fatal);

#endif // BC_ENABLE_LINE_LIB

/**
 * Frees a file, including flushing it.
 * @param f  The file to free.
 */
void
bc_file_free(BcFile* f);

/**
 * Print a char into the file.
 * @param f     The file to print to.
 * @param type  The flush type.
 * @param c     The character to write.
 */
void
bc_file_putchar(BcFile* restrict f, BcFlushType type, uchar c);

/**
 * Flush and return an error if it failed. This is meant to be used when needing
 * to flush in error situations when an error is already in flight. It would be
 * a very bad deal to throw another error.
 * @param f     The file to flush.
 * @param type  The flush type.
 * @return      A status indicating if an error occurred.
 */
BcStatus
bc_file_flushErr(BcFile* restrict f, BcFlushType type);

/**
 * Flush and throw an error on failure.
 * @param f     The file to flush.
 * @param type  The flush type.
 */
void
bc_file_flush(BcFile* restrict f, BcFlushType type);

/**
 * Write the contents of buf to the file.
 * @param f     The file to flush.
 * @param type  The flush type.
 * @param buf   The buffer whose contents will be written to the file.
 * @param n     The length of buf.
 */
void
bc_file_write(BcFile* restrict f, BcFlushType type, const char* buf, size_t n);

/**
 * Write to the file like fprintf would. This is very rudimentary.
 * @param f    The file to flush.
 * @param fmt  The format string.
 */
void
bc_file_printf(BcFile* restrict f, const char* fmt, ...);

/**
 * Write to the file like vfprintf would. This is very rudimentary.
 * @param f    The file to flush.
 * @param fmt  The format string.
 */
void
bc_file_vprintf(BcFile* restrict f, const char* fmt, va_list args);

/**
 * Write str to the file.
 * @param f     The file to flush.
 * @param type  The flush type.
 * @param str   The string to write to the file.
 */
void
bc_file_puts(BcFile* restrict f, BcFlushType type, const char* str);

#if BC_ENABLE_HISTORY && !BC_ENABLE_LINE_LIB

// Some constant flush types for ease of use.
extern const BcFlushType bc_flush_none;
extern const BcFlushType bc_flush_err;
extern const BcFlushType bc_flush_save;

#endif // BC_ENABLE_HISTORY && !BC_ENABLE_LINE_LIB

#endif // BC_FILE_H
