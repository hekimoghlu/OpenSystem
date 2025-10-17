/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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

/**
 * Convert the getaddrinfo() error code to a Kerberos et error code.
 *
 * @param eai_errno contains the error code from getaddrinfo().
 * @param system_error should have the value of errno after the failed getaddrinfo().
 *
 * @return Kerberos error code representing the EAI errors.
 *
 * @ingroup krb5_error
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_eai_to_heim_errno(int eai_errno, int system_error)
{
    switch(eai_errno) {
    case EAI_NOERROR:
	return 0;
#ifdef EAI_ADDRFAMILY
    case EAI_ADDRFAMILY:
	return HEIM_EAI_ADDRFAMILY;
#endif
    case EAI_AGAIN:
	return HEIM_EAI_AGAIN;
    case EAI_BADFLAGS:
	return HEIM_EAI_BADFLAGS;
    case EAI_FAIL:
	return HEIM_EAI_FAIL;
    case EAI_FAMILY:
	return HEIM_EAI_FAMILY;
    case EAI_MEMORY:
	return HEIM_EAI_MEMORY;
#if defined(EAI_NODATA) && EAI_NODATA != EAI_NONAME
    case EAI_NODATA:
	return HEIM_EAI_NODATA;
#endif
    case EAI_NONAME:
	return HEIM_EAI_NONAME;
    case EAI_SERVICE:
	return HEIM_EAI_SERVICE;
    case EAI_SOCKTYPE:
	return HEIM_EAI_SOCKTYPE;
#ifdef EAI_SYSTEM
    case EAI_SYSTEM:
	return system_error;
#endif
    default:
	return HEIM_EAI_UNKNOWN; /* XXX */
    }
}

/**
 * Convert the gethostname() error code (h_error) to a Kerberos et
 * error code.
 *
 * @param eai_errno contains the error code from gethostname().
 *
 * @return Kerberos error code representing the gethostname errors.
 *
 * @ingroup krb5_error
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_h_errno_to_heim_errno(int eai_errno)
{
    switch(eai_errno) {
    case 0:
	return 0;
    case HOST_NOT_FOUND:
	return HEIM_EAI_NONAME;
    case TRY_AGAIN:
	return HEIM_EAI_AGAIN;
    case NO_RECOVERY:
	return HEIM_EAI_FAIL;
    case NO_DATA:
	return HEIM_EAI_NONAME;
    default:
	return HEIM_EAI_UNKNOWN; /* XXX */
    }
}
