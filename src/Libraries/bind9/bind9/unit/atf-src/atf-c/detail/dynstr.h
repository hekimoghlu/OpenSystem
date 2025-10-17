/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
#if !defined(ATF_C_DYNSTR_H)
#define ATF_C_DYNSTR_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>

#include <atf-c/error_fwd.h>

/* ---------------------------------------------------------------------
 * The "atf_dynstr" type.
 * --------------------------------------------------------------------- */

struct atf_dynstr {
    char *m_data;
    size_t m_datasize;
    size_t m_length;
};
typedef struct atf_dynstr atf_dynstr_t;

/* Constants */
extern const size_t atf_dynstr_npos;

/* Constructors and destructors */
atf_error_t atf_dynstr_init(atf_dynstr_t *);
atf_error_t atf_dynstr_init_ap(atf_dynstr_t *, const char *, va_list);
atf_error_t atf_dynstr_init_fmt(atf_dynstr_t *, const char *, ...);
atf_error_t atf_dynstr_init_raw(atf_dynstr_t *, const void *, size_t);
atf_error_t atf_dynstr_init_rep(atf_dynstr_t *, size_t, char);
atf_error_t atf_dynstr_init_substr(atf_dynstr_t *, const atf_dynstr_t *,
                                   size_t, size_t);
atf_error_t atf_dynstr_copy(atf_dynstr_t *, const atf_dynstr_t *);
void atf_dynstr_fini(atf_dynstr_t *);
char *atf_dynstr_fini_disown(atf_dynstr_t *);

/* Getters */
const char *atf_dynstr_cstring(const atf_dynstr_t *);
size_t atf_dynstr_length(const atf_dynstr_t *);
size_t atf_dynstr_rfind_ch(const atf_dynstr_t *, char);

/* Modifiers */
atf_error_t atf_dynstr_append_ap(atf_dynstr_t *, const char *, va_list);
atf_error_t atf_dynstr_append_fmt(atf_dynstr_t *, const char *, ...);
void atf_dynstr_clear(atf_dynstr_t *);
atf_error_t atf_dynstr_prepend_ap(atf_dynstr_t *, const char *, va_list);
atf_error_t atf_dynstr_prepend_fmt(atf_dynstr_t *, const char *, ...);

/* Operators */
bool atf_equal_dynstr_cstring(const atf_dynstr_t *, const char *);
bool atf_equal_dynstr_dynstr(const atf_dynstr_t *, const atf_dynstr_t *);

#endif /* ATF_C_DYNSTR_H */
