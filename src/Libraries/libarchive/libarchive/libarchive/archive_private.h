/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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
#ifndef ARCHIVE_PRIVATE_H_INCLUDED
#define ARCHIVE_PRIVATE_H_INCLUDED

#ifndef __LIBARCHIVE_BUILD
#error This header is only to be used internally to libarchive.
#endif

#if HAVE_ICONV_H
#include <iconv.h>
#endif

#include "archive.h"
#include "archive_string.h"


#if defined(__GNUC__) && (__GNUC__ > 2 || \
						  (__GNUC__ == 2 && __GNUC_MINOR__ >= 5))
#define __LA_NORETURN __attribute__((__noreturn__))
#elif defined(_MSC_VER)
#define __LA_NORETURN __declspec(noreturn)
#else
#define __LA_NORETURN
#endif

#if defined(__GNUC__) && (__GNUC__ > 2 || \
			  (__GNUC__ == 2 && __GNUC_MINOR__ >= 7))
#define	__LA_UNUSED	__attribute__((__unused__))
#else
#define	__LA_UNUSED
#endif

#define	ARCHIVE_WRITE_MAGIC	(0xb0c5c0deU)
#define	ARCHIVE_READ_MAGIC	(0xdeb0c5U)
#define	ARCHIVE_WRITE_DISK_MAGIC (0xc001b0c5U)
#define	ARCHIVE_READ_DISK_MAGIC (0xbadb0c5U)
#define	ARCHIVE_MATCH_MAGIC	(0xcad11c9U)

#define	ARCHIVE_STATE_NEW	1U
#define	ARCHIVE_STATE_HEADER	2U
#define	ARCHIVE_STATE_DATA	4U
#define	ARCHIVE_STATE_EOF	0x10U
#define	ARCHIVE_STATE_CLOSED	0x20U
#define	ARCHIVE_STATE_FATAL	0x8000U
#define	ARCHIVE_STATE_ANY	(0xFFFFU & ~ARCHIVE_STATE_FATAL)

struct archive_vtable {
	int	(*archive_close)(struct archive *);
	int	(*archive_free)(struct archive *);
	int	(*archive_write_header)(struct archive *,
	    struct archive_entry *);
	int	(*archive_write_finish_entry)(struct archive *);
	ssize_t	(*archive_write_data)(struct archive *,
	    const void *, size_t);
	ssize_t	(*archive_write_data_block)(struct archive *,
	    const void *, size_t, int64_t);

	int	(*archive_read_next_header)(struct archive *,
	    struct archive_entry **);
	int	(*archive_read_next_header2)(struct archive *,
	    struct archive_entry *);
	int	(*archive_read_data_block)(struct archive *,
	    const void **, size_t *, int64_t *);

	int	(*archive_filter_count)(struct archive *);
	int64_t (*archive_filter_bytes)(struct archive *, int);
	int	(*archive_filter_code)(struct archive *, int);
	const char * (*archive_filter_name)(struct archive *, int);
};

struct archive_string_conv;

struct archive {
	/*
	 * The magic/state values are used to sanity-check the
	 * client's usage.  If an API function is called at a
	 * ridiculous time, or the client passes us an invalid
	 * pointer, these values allow me to catch that.
	 */
	unsigned int	magic;
	unsigned int	state;

	/*
	 * Some public API functions depend on the "real" type of the
	 * archive object.
	 */
	const struct archive_vtable *vtable;

	int		  archive_format;
	const char	 *archive_format_name;

	/* Number of file entries processed. */
	int		  file_count;

	int		  archive_error_number;
	const char	 *error;
	struct archive_string	error_string;

	char *current_code;
	unsigned current_codepage; /* Current ACP(ANSI CodePage). */
	unsigned current_oemcp; /* Current OEMCP(OEM CodePage). */
	struct archive_string_conv *sconv;

	/*
	 * Used by archive_read_data() to track blocks and copy
	 * data to client buffers, filling gaps with zero bytes.
	 */
	const char	 *read_data_block;
	int64_t		  read_data_offset;
	int64_t		  read_data_output_offset;
	size_t		  read_data_remaining;

	/*
	 * Used by formats/filters to determine the amount of data
	 * requested from a call to archive_read_data(). This is only
	 * useful when the format/filter has seek support.
	 */
	char		  read_data_is_posix_read;
	size_t		  read_data_requested;
};

/* Check magic value and state; return(ARCHIVE_FATAL) if it isn't valid. */
int	__archive_check_magic(struct archive *, unsigned int magic,
	    unsigned int state, const char *func);
#define	archive_check_magic(a, expected_magic, allowed_states, function_name) \
	do { \
		int magic_test = __archive_check_magic((a), (expected_magic), \
			(allowed_states), (function_name)); \
		if (magic_test == ARCHIVE_FATAL) \
			return ARCHIVE_FATAL; \
	} while (0)

__LA_NORETURN void	__archive_errx(int retvalue, const char *msg);

void	__archive_ensure_cloexec_flag(int fd);
int	__archive_mktemp(const char *tmpdir);
#if defined(_WIN32) && !defined(__CYGWIN__)
int	__archive_mkstemp(wchar_t *template);
#else
int	__archive_mkstemp(char *template);
#endif

int	__archive_clean(struct archive *);

void __archive_reset_read_data(struct archive *);

#define	err_combine(a,b)	((a) < (b) ? (a) : (b))

#if defined(__BORLANDC__) || (defined(_MSC_VER) &&  _MSC_VER <= 1300)
# define	ARCHIVE_LITERAL_LL(x)	x##i64
# define	ARCHIVE_LITERAL_ULL(x)	x##ui64
#else
# define	ARCHIVE_LITERAL_LL(x)	x##ll
# define	ARCHIVE_LITERAL_ULL(x)	x##ull
#endif

#endif
