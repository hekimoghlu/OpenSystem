/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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
/*
 *
 * Copyright(c) 2001-2017 Cisco Systems, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the following
 *   disclaimer in the documentation and/or other materials provided
 *   with the distribution.
 *
 *   Neither the name of the Cisco Systems, Inc. nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "err.h"
#include "datatypes.h"
#include <string.h>

/* srtp_err_file is the FILE to which errors are reported */

static FILE *srtp_err_file = NULL;

srtp_err_status_t srtp_err_reporting_init(void)
{
#ifdef ERR_REPORTING_STDOUT
    srtp_err_file = stdout;
#elif defined(ERR_REPORTING_FILE)
    /* open file for error reporting */
    srtp_err_file = fopen(ERR_REPORTING_FILE, "w");
    if (srtp_err_file == NULL) {
        return srtp_err_status_init_fail;
    }
#endif

    return srtp_err_status_ok;
}

static srtp_err_report_handler_func_t *srtp_err_report_handler = NULL;

srtp_err_status_t srtp_install_err_report_handler(
    srtp_err_report_handler_func_t func)
{
    srtp_err_report_handler = func;
    return srtp_err_status_ok;
}

void srtp_err_report(srtp_err_reporting_level_t level, const char *format, ...)
{
    char msg[512];
    va_list args;
    if (srtp_err_file != NULL) {
        va_start(args, format);
        vfprintf(srtp_err_file, format, args);
        va_end(args);
    }
    if (srtp_err_report_handler != NULL) {
        va_start(args, format);
        if (vsnprintf(msg, sizeof(msg), format, args) > 0) {
            /* strip trailing \n, callback should not have one */
            size_t l = strlen(msg);
            if (l && msg[l - 1] == '\n') {
                msg[l - 1] = '\0';
            }
            srtp_err_report_handler(level, msg);
            /*
             * NOTE, need to be carefull, there is a potential that
             * octet_string_set_to_zero() could
             * call srtp_err_report() in the future, leading to recursion
             */
            octet_string_set_to_zero(msg, sizeof(msg));
        }
        va_end(args);
    }
}
