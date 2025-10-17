/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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
#ifndef DST_PKCS11_H
#define DST_PKCS11_H 1

#include <isc/lang.h>
#include <isc/log.h>
#include <isc/result.h>

ISC_LANG_BEGINDECLS

isc_result_t
dst__pkcs11_toresult(const char *funcname, const char *file, int line,
		     isc_result_t fallback, CK_RV rv);

#define PK11_CALL(func, args, fallback) \
	((void) (((rv = (func) args) == CKR_OK) || \
		 ((ret = dst__pkcs11_toresult(#func, __FILE__, __LINE__, \
					      fallback, rv)), 0)))

#define PK11_RET(func, args, fallback) \
	((void) (((rv = (func) args) == CKR_OK) || \
		 ((ret = dst__pkcs11_toresult(#func, __FILE__, __LINE__, \
					      fallback, rv)), 0)));	\
	if (rv != CKR_OK) goto err;

ISC_LANG_ENDDECLS

#endif /* DST_PKCS11_H */
