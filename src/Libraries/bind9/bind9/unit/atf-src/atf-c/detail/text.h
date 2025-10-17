/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
#if !defined(ATF_C_TEXT_H)
#define ATF_C_TEXT_H

#include <stdarg.h>
#include <stdbool.h>

#include <atf-c/error_fwd.h>

#include "list.h"

atf_error_t atf_text_for_each_word(const char *, const char *,
                                   atf_error_t (*)(const char *, void *),
                                   void *);
atf_error_t atf_text_format(char **, const char *, ...);
atf_error_t atf_text_format_ap(char **, const char *, va_list);
atf_error_t atf_text_split(const char *, const char *, atf_list_t *);
atf_error_t atf_text_to_bool(const char *, bool *);
atf_error_t atf_text_to_long(const char *, long *);

#endif /* ATF_C_TEXT_H */
