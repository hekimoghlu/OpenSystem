/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
#include "kdc_locl.h"

#ifndef DEFAULT_KDC_LOG_DEST
#define DEFAULT_KDC_LOG_DEST "FILE:" KDC_LOG_DIR "/" KDC_LOG_FILE
#endif

void
kdc_openlog(krb5_context context,
	    const char *service,
	    krb5_kdc_configuration *config)
{
    char **s = NULL, **p;
    krb5_initlog(context, service, &config->logf);
    s = krb5_config_get_strings(context, NULL, service, "logging", NULL);
    if(s == NULL)
	s = krb5_config_get_strings(context, NULL, "logging", service, NULL);
    if(s){
	for(p = s; *p; p++)
	    krb5_addlog_dest(context, config->logf, *p);
	krb5_config_free_strings(s);
    }else {
	char *ss = NULL;
	asprintf(&ss, "0-10/%s", DEFAULT_KDC_LOG_DEST);
	krb5_addlog_dest(context, config->logf, ss);
	free(ss);
    }
    krb5_set_warn_dest(context, config->logf);
}

char*
kdc_log_msg_va(krb5_context context,
	       krb5_kdc_configuration *config,
	       int level, const char *fmt, va_list ap)
    HEIMDAL_PRINTF_ATTRIBUTE((printf, 4, 0))
{
    char *msg;
    krb5_vlog_msg(context, config->logf, &msg, level, fmt, ap);
    return msg;
}

char*
kdc_log_msg(krb5_context context,
	    krb5_kdc_configuration *config,
	    int level, const char *fmt, ...)
    HEIMDAL_PRINTF_ATTRIBUTE((printf, 4, 5))
{
    va_list ap;
    char *s;
    va_start(ap, fmt);
    s = kdc_log_msg_va(context, config, level, fmt, ap);
    va_end(ap);
    return s;
}

void
kdc_log(krb5_context context,
	krb5_kdc_configuration *config,
	int level, const char *fmt, ...)
    HEIMDAL_PRINTF_ATTRIBUTE((printf, 4, 5))
{
    va_list ap;
    char *s;
    va_start(ap, fmt);
    s = kdc_log_msg_va(context, config, level, fmt, ap);
    if(s) free(s);
    va_end(ap);
}
