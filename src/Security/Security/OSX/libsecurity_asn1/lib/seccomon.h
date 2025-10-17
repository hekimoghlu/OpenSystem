/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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
 * seccomon.h - common data structures for security libraries
 *
 * This file should have lowest-common-denominator datastructures
 * for security libraries.  It should not be dependent on any other
 * headers, and should not require linking with any libraries.
 *
 * $Id: seccomon.h,v 1.4 2004/03/23 21:31:41 mb Exp $
 */

#ifndef _SECCOMMON_H_
#define _SECCOMMON_H_

#include <security_asn1/prtypes.h>


#ifdef __cplusplus 
# define SEC_BEGIN_PROTOS extern "C" {
# define SEC_END_PROTOS }
#else
# define SEC_BEGIN_PROTOS
# define SEC_END_PROTOS
#endif

#include <security_asn1/secport.h>

#ifdef	__APPLE__

#include <Security/SecAsn1Types.h>

/*
 * Encode directly to/from SecAsn1Item.
 * Avoid the need for SECItemStr.type; see SEC_ANS1_SIGNED_INT
 * in secasn1t.h.
 */
typedef SecAsn1Item SECItem;
#else
/* Original NSS */
typedef enum {
    siBuffer = 0,
    siClearDataBuffer = 1,
    siCipherDataBuffer = 2,
    siDERCertBuffer = 3,
    siEncodedCertBuffer = 4,
    siDERNameBuffer = 5,
    siEncodedNameBuffer = 6,
    siAsciiNameString = 7,
    siAsciiString = 8,
    siDEROID = 9,
    siUnsignedInteger = 10,
    siUTCTime = 11,
    siGeneralizedTime = 12
} SECItemType;

typedef struct SECItemStr SECItem;

struct SECItemStr {
    SECItemType type;
    unsigned char *data;
    unsigned int len;
};
#endif	/* __APPLE__ */

/*
** A status code. Status's are used by procedures that return status
** values. Again the motivation is so that a compiler can generate
** warnings when return values are wrong. Correct testing of status codes:
**
**	SECStatus rv;
**	rv = some_function (some_argument);
**	if (rv != SECSuccess)
**		do_an_error_thing();
**
*/
typedef enum _SECStatus {
    SECWouldBlock = -2,
    SECFailure = -1,
    SECSuccess = 0
} SECStatus;

/*
** A comparison code. Used for procedures that return comparision
** values. Again the motivation is so that a compiler can generate
** warnings when return values are wrong.
*/
typedef enum _SECComparison {
    SECLessThan = -1,
    SECEqual = 0,
    SECGreaterThan = 1
} SECComparison;

#endif /* _SECCOMMON_H_ */
