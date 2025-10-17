/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#ifdef PKCS11CRYPTO

#include <config.h>

#include <dns/log.h>
#include <dns/result.h>

#include <pk11/pk11.h>
#include <pk11/internal.h>

#include "dst_pkcs11.h"

isc_result_t
dst__pkcs11_toresult(const char *funcname, const char *file, int line,
		     isc_result_t fallback, CK_RV rv)
{
	isc_log_write(dns_lctx, DNS_LOGCATEGORY_GENERAL,
		      DNS_LOGMODULE_CRYPTO, ISC_LOG_WARNING,
		      "%s:%d: %s: Error = 0x%.8lX\n",
		      file, line, funcname, rv);
	if (rv == CKR_HOST_MEMORY)
		return (ISC_R_NOMEMORY);
	return (fallback);
}


#else /* PKCS11CRYPTO */

#include <isc/util.h>

EMPTY_TRANSLATION_UNIT

#endif /* PKCS11CRYPTO */
/*! \file */
