/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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
#include "gsskrb5_locl.h"

#undef HEIMDAL_PRINTF_ATTRIBUTE
#define HEIMDAL_PRINTF_ATTRIBUTE(x)

static const char *
calling_error(OM_uint32 v)
{
    static const char *msgs[] = {
	NULL,			/* 0 */
	"A required input parameter could not be read.", /*  */
	"A required output parameter could not be written.", /*  */
	"A parameter was malformed"
    };

    v >>= GSS_C_CALLING_ERROR_OFFSET;

    if (v == 0)
	return "";
    else if (v >= sizeof(msgs)/sizeof(*msgs))
	return "unknown calling error";
    else
	return msgs[v];
}

static const char *
routine_error(OM_uint32 v)
{
    static const char *msgs[] = {
	NULL,			/* 0 */
	"An unsupported mechanism was requested",
	"An invalid name was supplied",
	"A supplied name was of an unsupported type",
	"Incorrect channel bindings were supplied",
	"An invalid status code was supplied",
	"A token had an invalid MIC",
	"No credentials were supplied, "
	"or the credentials were unavailable or inaccessible.",
	"No context has been established",
	"A token was invalid",
	"A credential was invalid",
	"The referenced credentials have expired",
	"The context has expired",
	"Miscellaneous failure (see text)",
	"The quality-of-protection requested could not be provide",
	"The operation is forbidden by local security policy",
	"The operation or option is not available",
	"The requested credential element already exists",
	"The provided name was not a mechanism name.",
    };

    v >>= GSS_C_ROUTINE_ERROR_OFFSET;

    if (v == 0)
	return "";
    else if (v >= sizeof(msgs)/sizeof(*msgs))
	return "unknown routine error";
    else
	return msgs[v];
}

static const char *
supplementary_error(OM_uint32 v)
{
    static const char *msgs[] = {
	"normal completion",
	"continuation call to routine required",
	"duplicate per-message token detected",
	"timed-out per-message token detected",
	"reordered (early) per-message token detected",
	"skipped predecessor token(s) detected"
    };

    v >>= GSS_C_SUPPLEMENTARY_OFFSET;

    if (v >= sizeof(msgs)/sizeof(*msgs))
	return "unknown routine error";
    else
	return msgs[v];
}

void
_gsskrb5_clear_status (void)
{
    krb5_context context;

    if (_gsskrb5_init (&context) != 0)
	return;
    krb5_clear_error_message(context);
}

void
_gsskrb5_set_status (int ret, const char *fmt, ...)
    HEIMDAL_PRINTF_ATTRIBUTE((printf, 2, 3))
{
    krb5_context context;
    va_list args;
    char *str;
    int e;

    if (_gsskrb5_init (&context) != 0)
	return;

    va_start(args, fmt);
    e = vasprintf(&str, fmt, args);
    va_end(args);
    if (e >= 0 && str) {
	krb5_set_error_message(context, ret, "%s", str);
	free(str);
    }
}

OM_uint32 GSSAPI_CALLCONV _gsskrb5_display_status
(OM_uint32		*minor_status,
 OM_uint32		 status_value,
 int			 status_type,
 const gss_OID	 mech_type,
 OM_uint32		*message_context,
 gss_buffer_t	 status_string)
{
    krb5_context context;
    char *buf = NULL;
    int e = 0;

    GSSAPI_KRB5_INIT (&context);

    status_string->length = 0;
    status_string->value = NULL;

    if (gss_oid_equal(mech_type, GSS_C_NO_OID) == 0 &&
	gss_oid_equal(mech_type, GSS_KRB5_MECHANISM) == 0) {
	*minor_status = 0;
	return GSS_C_GSS_CODE;
    }

    if (status_type == GSS_C_GSS_CODE) {
	if (GSS_SUPPLEMENTARY_INFO(status_value))
	    e = asprintf(&buf, "%s",
			 supplementary_error(GSS_SUPPLEMENTARY_INFO(status_value)));
	else
	    e = asprintf (&buf, "%s %s",
			  calling_error(GSS_CALLING_ERROR(status_value)),
			  routine_error(GSS_ROUTINE_ERROR(status_value)));
    } else if (status_type == GSS_C_MECH_CODE) {
	const char *buf2 = krb5_get_error_message(context, status_value);
	if (buf2) {
	    buf = strdup(buf2);
	    krb5_free_error_message(context, buf2);
	} else {
	    e = asprintf(&buf, "unknown mech error-code %u",
			 (unsigned)status_value);
	}
    } else {
	*minor_status = EINVAL;
	return GSS_S_BAD_STATUS;
    }

    if (e < 0 || buf == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    *message_context = 0;
    *minor_status = 0;

    status_string->length = strlen(buf);
    status_string->value  = buf;

    return GSS_S_COMPLETE;
}
