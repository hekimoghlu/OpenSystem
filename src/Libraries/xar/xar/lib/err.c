/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
 * 03-Apr-2005
 * DRI: Rob Braun <bbraun@synack.net>
 */

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "xar.h"
#include "archive.h"

#define ECTX(x) ((struct errctx *)(x))

void xar_register_errhandler(xar_t x, err_handler callback, void *usrctx) {
	ECTX(&XAR(x)->errctx)->x = x;
	ECTX(&XAR(x)->errctx)->usrctx = usrctx;
	XAR(x)->ercallback = callback;
	return;
}

xar_t xar_err_get_archive(xar_errctx_t ctx) {
	return ECTX(ctx)->x;
}

xar_file_t xar_err_get_file(xar_errctx_t ctx) {
	return ECTX(ctx)->file;
}

void  xar_err_set_file(xar_t x, xar_file_t f) {
	XAR(x)->errctx.file = f;
	return;
}

const char *xar_err_get_string(xar_errctx_t ctx) {
	return ECTX(ctx)->str;
}

void xar_err_set_string(xar_t x, const char *str) {
	XAR(x)->errctx.str = strdup(str); // this leaks right now, but it's safer than the alternative
	return;
}

void xar_err_set_formatted_string(xar_t x, const char *format, ...) {
	va_list arg;
	char *msg;
	va_start(arg, format);
	vasprintf(&msg, format, arg);
	va_end(arg);
	xar_err_set_string(x, msg);
	free(msg);
}


int xar_err_get_errno(xar_errctx_t ctx) {
	return ECTX(ctx)->saved_errno;
}

void  xar_err_set_errno(xar_t x, int e) {
	XAR(x)->errctx.saved_errno = e;
	return;
}

void xar_err_new(xar_t x) {
	memset(&XAR(x)->errctx, 0, sizeof(struct errctx));
	XAR(x)->errctx.saved_errno = errno;
	return;
}

int32_t xar_err_callback(xar_t x, int32_t sev, int32_t err) {
	if( XAR(x)->ercallback )
		return XAR(x)->ercallback(sev, err, &XAR(x)->errctx, ECTX(&XAR(x)->errctx)->usrctx);
	return 0;
}
