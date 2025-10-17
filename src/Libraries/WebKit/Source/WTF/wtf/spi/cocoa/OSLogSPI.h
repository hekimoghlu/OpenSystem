/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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

#include <os/log.h>

#if USE(APPLE_INTERNAL_SDK)

#include <os/log_private.h>

#else

typedef uint32_t os_trace_mode_t;

OS_ENUM(_os_trace_commonmodes, os_trace_mode_t,
    OS_TRACE_MODE_INFO                          = 0x01,
    OS_TRACE_MODE_DEBUG                         = 0x02,
    OS_TRACE_MODE_BACKTRACE                     = 0x04,
    OS_TRACE_MODE_STREAM_LIVE                   = 0x08,
    OS_TRACE_MODE_DISABLE                       = 0x0100,
    OS_TRACE_MODE_OFF                           = 0x0400,
);

typedef struct os_log_message_s {
    uint64_t trace_id;
    char padding[80];
    const char *format;
    const uint8_t *buffer;
    size_t buffer_sz;
    const uint8_t *privdata;
    size_t privdata_sz;
    const char *subsystem;
    const char *category;
} *os_log_message_t;

#endif

WTF_EXTERN_C_BEGIN

OS_EXPORT OS_NOTHROW OS_NOT_TAIL_CALLED OS_NONNULL5
void os_log_with_args(os_log_t oslog, os_log_type_t type, const char *format, va_list args, void *ret_addr);

OS_EXPORT OS_NOTHROW
void os_trace_set_mode(os_trace_mode_t mode);

OS_EXPORT OS_NOTHROW
os_trace_mode_t os_trace_get_mode();

typedef void (^os_log_hook_t)(os_log_type_t type, os_log_message_t msg);

OS_EXPORT OS_NOTHROW
os_log_hook_t os_log_set_hook(os_log_type_t level, os_log_hook_t);

OS_EXPORT OS_NOTHROW
char* os_log_copy_message_string(os_log_message_t msg);

WTF_EXTERN_C_END
