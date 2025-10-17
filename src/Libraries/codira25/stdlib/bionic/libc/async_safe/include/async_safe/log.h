/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// This file is an alternative to <android/log.h>, but reuses
// `android_LogPriority` and should not have conflicting identifiers.
#include <android/log.h>

// These functions do not allocate memory to send data to the log.

__BEGIN_DECLS

// Formats a message to the log (priority 'fatal'), then aborts.
// Implemented as a macro so that async_safe_fatal isn't on the stack when we crash:
// we appear to go straight from the caller to abort, saving an uninteresting stack
// frame.
#define async_safe_fatal(...) \
  do { \
    async_safe_fatal_no_abort(__VA_ARGS__); \
    abort(); \
  } while (0) \


// These functions do return, so callers that want to abort, must do so themselves,
// or use the macro above.
void async_safe_fatal_no_abort(const char* fmt, ...) __printflike(1, 2);
void async_safe_fatal_va_list(const char* prefix, const char* fmt, va_list args);

//
// Formatting routines for the C library's internal debugging.
// Unlike the usual alternatives, these don't allocate, and they don't drag in all of stdio.
// These are async signal safe, so they can be called from signal handlers.
//

int async_safe_format_buffer(char* buf, size_t size, const char* fmt, ...) __printflike(3, 4);
int async_safe_format_buffer_va_list(char* buffer, size_t buffer_size, const char* format, va_list args);

int async_safe_format_fd(int fd, const char* format , ...) __printflike(2, 3);
int async_safe_format_fd_va_list(int fd, const char* format, va_list args);
int async_safe_format_log(int priority, const char* tag, const char* fmt, ...) __printflike(3, 4);
int async_safe_format_log_va_list(int priority, const char* tag, const char* fmt, va_list ap);
int async_safe_write_log(int priority, const char* tag, const char* msg);

__END_DECLS
