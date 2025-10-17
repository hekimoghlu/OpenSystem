/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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

#if USE(OS_STATE)

#if USE(APPLE_INTERNAL_SDK)
#include <os/state_private.h>
#else

#include <os/base.h>

OS_ENUM(os_state_reason, uint32_t,
    OS_STATE_REASON_GENERAL          = 0x0001,
    OS_STATE_REASON_NETWORKING       = 0x0002,
    OS_STATE_REASON_CELLULAR         = 0x0004,
    OS_STATE_REASON_AUTHENTICATION   = 0x0008,
);

OS_ENUM(os_state_api, uint32_t,
    OS_STATE_API_ERROR = 1,
    OS_STATE_API_FAULT = 2,
    OS_STATE_API_REQUEST = 3,
);

OS_ENUM(os_state_data_type, uint32_t,
    OS_STATE_DATA_SERIALIZED_NSCF_OBJECT = 1,
    OS_STATE_DATA_PROTOCOL_BUFFER = 2,
    OS_STATE_DATA_CUSTOM = 3,
);

typedef struct os_state_hints_s {
    uint32_t osh_version;
    const char *osh_requestor;
    os_state_api_t osh_api;
    os_state_reason_t osh_reason;
} *os_state_hints_t;

typedef struct os_state_data_decoder_s {
    char osdd_library[64];
    char osdd_type[64];
} *os_state_data_decoder_t;

typedef struct os_state_data_s {
    os_state_data_type_t osd_type;
    IGNORE_CLANG_WARNINGS_BEGIN("packed")
    union {
        uint64_t osd_size:32;
        uint32_t osd_data_size;
    } __attribute__((packed, aligned(4)));
    IGNORE_CLANG_WARNINGS_END
    struct os_state_data_decoder_s osd_decoder;
    char osd_title[64];
    uint8_t osd_data[];
} *os_state_data_t;

typedef uint64_t os_state_handle_t;
typedef os_state_data_t (^os_state_block_t)(os_state_hints_t hints);

WTF_EXTERN_C_BEGIN

OS_EXPORT OS_NOTHROW OS_NOT_TAIL_CALLED
os_state_handle_t
os_state_add_handler(dispatch_queue_t, os_state_block_t);

WTF_EXTERN_C_END

#define OS_STATE_DATA_SIZE_NEEDED(data_size) (sizeof(struct os_state_data_s) + data_size)

#endif

#endif // USE(OS_STATE)
