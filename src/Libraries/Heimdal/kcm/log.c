/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
#include "kcm_locl.h"

static krb5_log_facility *lfac;

void
kcm_openlog(void)
{
    char **s = NULL, **p;
    krb5_initlog(kcm_context, "kcm", &lfac);
    s = krb5_config_get_strings(kcm_context, NULL, "kcm", "logging", NULL);
    if(s == NULL)
	s = krb5_config_get_strings(kcm_context, NULL, "logging", "kcm", NULL);
    if(s){
	for(p = s; *p; p++)
	    krb5_addlog_dest(kcm_context, lfac, *p);
	krb5_config_free_strings(s);
    }else
	krb5_addlog_dest(kcm_context, lfac, DEFAULT_LOG_DEST);
    krb5_set_warn_dest(kcm_context, lfac);
}

#undef __attribute__
#define __attribute__(X)

char*
kcm_log_msg_va(int level, const char *fmt, va_list ap)
     __attribute__ ((format (printf, 2, 0)))
{
    char *msg;
    krb5_vlog_msg(kcm_context, lfac, &msg, level, fmt, ap);
    return msg;
}

char*
kcm_log_msg(int level, const char *fmt, ...)
     __attribute__ ((format (printf, 2, 3)))
{
    va_list ap;
    char *s;
    va_start(ap, fmt);
    s = kcm_log_msg_va(level, fmt, ap);
    va_end(ap);
    return s;
}

void
kcm_log(int level, const char *fmt, ...)
     __attribute__ ((format (printf, 2, 3)))
{
    va_list ap;
    char *s;
    va_start(ap, fmt);
    s = kcm_log_msg_va(level, fmt, ap);
    if(s) free(s);
    va_end(ap);
}
