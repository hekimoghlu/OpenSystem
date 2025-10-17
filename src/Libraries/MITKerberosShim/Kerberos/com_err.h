/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
/* $Header$ */

/* 
 *
 * KerberosComErr.h -- Functions to handle Kerberos errors.
 *
 */


#ifndef __KERBEROSCOMERR__
#define __KERBEROSCOMERR__

#ifndef __has_extension
#define __has_extension(x) 0
#endif

#ifndef KERBEROS_APPLE_DEPRECATED
#if __has_extension(attribute_deprecated_with_message)
#define KERBEROS_APPLE_DEPRECATED(x) __attribute__((deprecated(x)))
#else
#if !defined(__GNUC__) && !defined(__attribute__)
#define __attribute__(x)
#endif
#define KERBEROS_APPLE_DEPRECATED(x) __attribute__((deprecated))
#endif
#endif

#if defined(macintosh) || (defined(__MACH__) && defined(__APPLE__))
#    include <TargetConditionals.h>
#    if TARGET_RT_MAC_CFM
#        error "Use KfM 4.0 SDK headers for CFM compilation."
#    endif
#endif

#include <sys/types.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef long errcode_t;
typedef void (*com_err_handler_t)
    (const char *whoami, errcode_t code, const char *format, va_list args);

struct error_table {
    const char * const * const messages;
    int32_t base;
    int32_t count;
};

/* ******************* */
/* Function prototypes */
/* ******************* */

void com_err    (const char *progname, errcode_t code, const char *format, ...);
void com_err_va (const char *progname, errcode_t code, const char *format, va_list args);

const char *error_message (errcode_t code);

com_err_handler_t set_com_err_hook(com_err_handler_t handler);
com_err_handler_t reset_com_err_hook(void);

errcode_t add_error_table    (const struct error_table *et);
errcode_t remove_error_table (const struct error_table *et);

#ifdef __cplusplus
}
#endif

#endif /* __KERBEROSCOMERR__ */
