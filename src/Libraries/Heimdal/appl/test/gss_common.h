/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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
/* $Id$ */

void write_token (int sock, gss_buffer_t buf);
void read_token (int sock, gss_buffer_t buf);

void gss_print_errors (int min_stat);

void gss_verr(int exitval, int status, const char *fmt, va_list ap)
    __attribute__ ((format (printf, 3, 0)));

void gss_err(int exitval, int status, const char *fmt, ...)
    __attribute__ ((format (printf, 3, 4)));

gss_OID select_mech(const char *);

void print_gss_name(const char *, gss_name_t);
