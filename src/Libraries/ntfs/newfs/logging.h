/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
#ifndef _LOGGING_H_
#define _LOGGING_H_

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif

#include "types.h"

/* Function prototype for the logging handlers */
typedef int (ntfs_log_handler)(const char *function, const char *file, int line,
	u32 level, void *data, const char *format, va_list args);

/* Set the logging handler from one of the functions, below. */
void ntfs_log_set_handler(ntfs_log_handler *handler 
			  __attribute__((format(printf, 6, 0))));

/* Logging handlers */
ntfs_log_handler ntfs_log_handler_syslog  __attribute__((format(printf, 6, 0)));
ntfs_log_handler ntfs_log_handler_fprintf __attribute__((format(printf, 6, 0)));
ntfs_log_handler ntfs_log_handler_null    __attribute__((format(printf, 6, 0)));
ntfs_log_handler ntfs_log_handler_outerr  __attribute__((format(printf, 6, 0)));
ntfs_log_handler ntfs_log_handler_outerr_syslog
                                          __attribute__((format(printf, 6, 0)));

/* Enable/disable certain log levels */
u32 ntfs_log_set_levels(u32 levels);
u32 ntfs_log_clear_levels(u32 levels);

/* Turn command-line options into logging flags */
BOOL ntfs_log_parse_option(const char *option);

int ntfs_log_redirect(const char *function, const char *file, int line,
	u32 level, void *data, const char *format, ...)
	__attribute__((format(printf, 6, 7)));

/* Logging levels - Determine what gets logged */
#define NTFS_LOG_LEVEL_DEBUG	(1 <<  0) /* x = 42 */
#define NTFS_LOG_LEVEL_TRACE	(1 <<  1) /* Entering function x() */
#define NTFS_LOG_LEVEL_QUIET	(1 <<  2) /* Quietable output */
#define NTFS_LOG_LEVEL_INFO	(1 <<  3) /* Volume needs defragmenting */
#define NTFS_LOG_LEVEL_VERBOSE	(1 <<  4) /* Forced to continue */
#define NTFS_LOG_LEVEL_PROGRESS	(1 <<  5) /* 54% complete */
#define NTFS_LOG_LEVEL_WARNING	(1 <<  6) /* You should backup before starting */
#define NTFS_LOG_LEVEL_ERROR	(1 <<  7) /* Operation failed, no damage done */
#define NTFS_LOG_LEVEL_PERROR	(1 <<  8) /* Message : standard error description */
#define NTFS_LOG_LEVEL_CRITICAL	(1 <<  9) /* Operation failed,damage may have occurred */
#define NTFS_LOG_LEVEL_ENTER	(1 << 10) /* Enter a function */
#define NTFS_LOG_LEVEL_LEAVE	(1 << 11) /* Leave a function  */

/* Logging style flags - Manage the style of the output */
#define NTFS_LOG_FLAG_PREFIX	(1 << 0) /* Prefix messages with "ERROR: ", etc */
#define NTFS_LOG_FLAG_FILENAME	(1 << 1) /* Show the file origin of the message */
#define NTFS_LOG_FLAG_LINE	(1 << 2) /* Show the line number of the message */
#define NTFS_LOG_FLAG_FUNCTION	(1 << 3) /* Show the function name containing the message */
#define NTFS_LOG_FLAG_ONLYNAME	(1 << 4) /* Only display the filename, not the pathname */

/* Macros to simplify logging.  One for each level defined above.
 * Note, ntfs_log_debug/trace have effect only if DEBUG is defined.
 */
#define ntfs_log_critical(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_CRITICAL,NULL,FORMAT,##ARGS)
#define ntfs_log_error(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_ERROR,NULL,FORMAT,##ARGS)
#define ntfs_log_info(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_INFO,NULL,FORMAT,##ARGS)
#define ntfs_log_perror(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_PERROR,NULL,FORMAT,##ARGS)
#define ntfs_log_progress(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_PROGRESS,NULL,FORMAT,##ARGS)
#define ntfs_log_quiet(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_QUIET,NULL,FORMAT,##ARGS)
#define ntfs_log_verbose(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_VERBOSE,NULL,FORMAT,##ARGS)
#define ntfs_log_warning(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_WARNING,NULL,FORMAT,##ARGS)

/* By default debug and trace messages are compiled into the program,
 * but not displayed.
 */
#ifdef DEBUG
#define ntfs_log_debug(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_DEBUG,NULL,FORMAT,##ARGS)
#define ntfs_log_trace(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_TRACE,NULL,FORMAT,##ARGS)
#define ntfs_log_enter(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_ENTER,NULL,FORMAT,##ARGS)
#define ntfs_log_leave(FORMAT, ARGS...) ntfs_log_redirect(__FUNCTION__,__FILE__,__LINE__,NTFS_LOG_LEVEL_LEAVE,NULL,FORMAT,##ARGS)
#else
#define ntfs_log_debug(FORMAT, ARGS...)do {} while (0)
#define ntfs_log_trace(FORMAT, ARGS...)do {} while (0)
#define ntfs_log_enter(FORMAT, ARGS...)do {} while (0)
#define ntfs_log_leave(FORMAT, ARGS...)do {} while (0)
#endif /* DEBUG */

#endif /* _LOGGING_H_ */
