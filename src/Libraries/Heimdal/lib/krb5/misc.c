/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
#include "krb5_locl.h"
#ifdef HAVE_EXECINFO_H
#include <execinfo.h>
#endif

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
_krb5_s4u2self_to_checksumdata(krb5_context context,
			       const PA_S4U2Self *self,
			       krb5_data *data)
{
    krb5_error_code ret;
    krb5_ssize_t ssize;
    krb5_storage *sp;
    size_t size;
    size_t i;

    sp = krb5_storage_emem();
    if (sp == NULL) {
	krb5_clear_error_message(context);
	return ENOMEM;
    }
    krb5_storage_set_flags(sp, KRB5_STORAGE_BYTEORDER_LE);
    ret = krb5_store_int32(sp, self->name.name_type);
    if (ret)
	goto out;
    for (i = 0; i < self->name.name_string.len; i++) {
	size = strlen(self->name.name_string.val[i]);
	ssize = krb5_storage_write(sp, self->name.name_string.val[i], size);
	if (ssize != (krb5_ssize_t)size) {
	    ret = ENOMEM;
	    goto out;
	}
    }
    size = strlen(self->realm);
    ssize = krb5_storage_write(sp, self->realm, size);
    if (ssize != (krb5_ssize_t)size) {
	ret = ENOMEM;
	goto out;
    }
    size = strlen(self->auth);
    ssize = krb5_storage_write(sp, self->auth, size);
    if (ssize != (krb5_ssize_t)size) {
	ret = ENOMEM;
	goto out;
    }

    ret = krb5_storage_to_data(sp, data);
    krb5_storage_free(sp);
    return ret;

out:
    krb5_clear_error_message(context);
    return ret;
}

krb5_error_code
krb5_enomem(krb5_context context)
{
    krb5_set_error_message(context, ENOMEM, N_("malloc: out of memory", ""));
    return ENOMEM;
}

void
_krb5_debug_backtrace(krb5_context context)
{
#if defined(HAVE_BACKTRACE) && !defined(HEIMDAL_SMALLER)
    void *stack[128];
    char **strs = NULL;
    int i, frames = backtrace(stack, sizeof(stack) / sizeof(stack[0]));
    if (frames > 0)
	strs = backtrace_symbols(stack, frames);
    if (strs) {
	for (i = 0; i < frames; i++)
	    _krb5_debugx(context, 10, "frame %d: %s", i, strs[i]);
	free(strs);
    }
#endif
}

krb5_error_code
_krb5_einval(krb5_context context, const char *func, unsigned long argn)
{
#ifndef HEIMDAL_SMALLER
    krb5_set_error_message(context, EINVAL,
			   N_("programmer error: invalid argument to %s argument %lu",
			      "function:line"),
			   func, argn);
    if (_krb5_have_debug(context, 10)) {
	_krb5_debugx(context, 10, "invalid argument to function %s argument %lu",
		     func, argn);
	_krb5_debug_backtrace(context);
    }
#endif
    return EINVAL;
}

static const char hexchar[16] = "0123456789ABCDEF";

char * KRB5_LIB_FUNCTION
krb5_uuid_to_string(krb5_uuid uuid)
{
    char *string, *p;
    size_t n;

    string = malloc((sizeof(krb5_uuid) * 2) + 5);
    if (string == NULL)
	return NULL;

    for (n = 0, p = string; n < sizeof(krb5_uuid); n++) {
	if (n == 4 || n == 6 || n == 8 || n == 10)
	    *p++ = '-';
	*p++ = hexchar[uuid[n] >> 4];
	*p++ = hexchar[uuid[n] & 0xf];
    }
    *p = '\0';
    return string;
}

static int
pos(char c)
{
    const char *p;
    c = toupper((unsigned char)c);
    for (p = hexchar; *p; p++)
	if (*p == c)
	    return (int)(p - hexchar);
    return -1;
}

krb5_error_code KRB5_LIB_FUNCTION
krb5_string_to_uuid(const char *str, krb5_uuid uuid)
{
    size_t n;

    if (strlen(str) != 36)
	return EINVAL;

    for (n = 0; n < sizeof(krb5_uuid); n++) {
	if (n == 4 || n == 6 || n == 8 || n == 10) {
	    if (*str++ != '-')
		return EINVAL;
	}
	uuid[n] = pos(str[0]) << 4 | pos(str[1]);
	str += 2;
    }
    return 0;
}
