/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
#if !defined(ATF_C_ERROR_H)
#define ATF_C_ERROR_H

#include <stdbool.h>
#include <stddef.h>

#include <atf-c/error_fwd.h>

/* ---------------------------------------------------------------------
 * The "atf_error" type.
 * --------------------------------------------------------------------- */

struct atf_error {
    bool m_free;
    const char *m_type;
    void *m_data;

    void (*m_format)(struct atf_error *, char *, size_t);
};

atf_error_t atf_error_new(const char *, void *, size_t,
                          void (*)(const atf_error_t, char *, size_t));
void atf_error_free(atf_error_t);

atf_error_t atf_no_error(void);
bool atf_is_error(const atf_error_t);

bool atf_error_is(const atf_error_t, const char *);
const void *atf_error_data(const atf_error_t);
void atf_error_format(const atf_error_t, char *, size_t);

/* ---------------------------------------------------------------------
 * Common error types.
 * --------------------------------------------------------------------- */

atf_error_t atf_libc_error(int, const char *, ...);
int atf_libc_error_code(const atf_error_t);
const char *atf_libc_error_msg(const atf_error_t);

atf_error_t atf_no_memory_error(void);

#endif /* ATF_C_ERROR_H */
