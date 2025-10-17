/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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
#include "hx_locl.h"

#undef HEIMDAL_PRINTF_ATTRIBUTE
#define HEIMDAL_PRINTF_ATTRIBUTE(x)

/**
 * @page page_error Hx509 error reporting functions
 *
 * See the library functions here: @ref hx509_error
 */

struct hx509_error_data {
    hx509_error next;
    int code;
    char *msg;
};

/**
 * Resets the error strings the hx509 context.
 *
 * @param context A hx509 context.
 *
 * @ingroup hx509_error
 */

void
hx509_clear_error_string(hx509_context context)
{
    if (context) {
	heim_release(context->error);
	context->error = NULL;
    }
}

/**
 * Add an error message to the hx509 context.
 *
 * @param context A hx509 context.
 * @param flags
 * - HX509_ERROR_APPEND appends the error string to the old messages
     (code is updated).
 * @param code error code related to error message
 * @param fmt error message format
 * @param ap arguments to error message format
 *
 * @ingroup hx509_error
 */

void
hx509_set_error_stringv(hx509_context context, int flags, int code,
			const char *fmt, va_list ap)
    HEIMDAL_PRINTF_ATTRIBUTE((printf, 4, 0))
{
    heim_error_t msg;

    if (context == NULL)
	return;

    msg = heim_error_createv(code, fmt, ap);
    if (msg) {
	if (flags & HX509_ERROR_APPEND)
	    heim_error_append(msg, context->error);
	heim_release(context->error);
    }
    context->error = msg;
}

/**
 * See hx509_set_error_stringv().
 *
 * @param context A hx509 context.
 * @param flags
 * - HX509_ERROR_APPEND appends the error string to the old messages
     (code is updated).
 * @param code error code related to error message
 * @param fmt error message format
 * @param ... arguments to error message format
 *
 * @ingroup hx509_error
 */

void
hx509_set_error_string(hx509_context context, int flags, int code,
		       const char *fmt, ...)
    HEIMDAL_PRINTF_ATTRIBUTE((printf, 4, 5))
{
    va_list ap;

    va_start(ap, fmt);
    hx509_set_error_stringv(context, flags, code, fmt, ap);
    va_end(ap);
}

/**
 * Get an error string from context associated with error_code.
 *
 * @param context A hx509 context.
 * @param error_code Get error message for this error code.
 *
 * @return error string, free with hx509_free_error_string().
 *
 * @ingroup hx509_error
 */

char *
hx509_get_error_string(hx509_context context, int error_code)
{
    heim_error_t msg = context->error;
    heim_string_t s;
    char *str = NULL;
    const char *cstr;

    if (msg == NULL || heim_error_get_code(msg) != error_code) {
	char buf[256];

	cstr = com_right_r(context->et_list, error_code, buf, sizeof(buf));
	if (cstr)
	    return strdup(cstr);
	cstr = strerror(error_code);
	if (cstr)
	    return strdup(cstr);
	if (asprintf(&str, "<unknown error: %d>", error_code) == -1)
	    return NULL;
	return str;
    }

    s = heim_error_copy_string(msg);
    if (s) {
	str = heim_string_copy_utf8(s);
	heim_release(s);
    }
    return str;
}

/**
 * Free error string returned by hx509_get_error_string().
 *
 * @param str error string to free.
 *
 * @ingroup hx509_error
 */

void
hx509_free_error_string(char *str)
{
    free(str);
}

/**
 * Print error message and fatally exit from error code
 *
 * @param context A hx509 context.
 * @param exit_code exit() code from process.
 * @param error_code Error code for the reason to exit.
 * @param fmt format string with the exit message.
 * @param ... argument to format string.
 *
 * @ingroup hx509_error
 */

void
hx509_err(hx509_context context, int exit_code,
	  int error_code, const char *fmt, ...)
    HEIMDAL_PRINTF_ATTRIBUTE((printf, 4, 5))
    HEIMDAL_NORETURN_ATTRIBUTE
{
    va_list ap;
    const char *msg;
    char *str;

    va_start(ap, fmt);
    vasprintf(&str, fmt, ap);
    va_end(ap);
    msg = hx509_get_error_string(context, error_code);
    if (msg == NULL)
	msg = "no error";

    errx(exit_code, "%s: %s", str, msg);
}
