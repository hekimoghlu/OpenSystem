/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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
#ifndef	_CC_CMACSPI_H_
#define _CC_CMACSPI_H_

#include <stdint.h>
#include <sys/types.h>

#if defined(_MSC_VER)
#include <availability.h>
#else
#include <os/availability.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

#define CC_CMACAES_DIGEST_LENGTH     16          /* CMAC length in bytes - copy and paste error - */

#define CC_CMACAES_OUTPUT_LENGTH     16          /* CMAC length in bytes */

/*!
    @function   CCAESCmac
    @abstract   Stateless, one-shot AES CMAC function

    @param      key         Raw key bytes.
    @param      data        The data to process.
    @param      dataLength  The length of the data to process.
    @param      macOut      The MAC bytes (space provided by the caller).
                            Output is written to caller-supplied buffer.

    @discussion The length of the MAC written to *macOut is 16
                The MAC must be verified by comparing the computed and expected values
                using timingsafe_bcmp. Other comparison functions (e.g. memcmp)
                must not be used as they may be vulnerable to practical timing attacks,
                leading to MAC forgery.
*/

void
CCAESCmac(const void *key, const uint8_t *data, size_t dataLength, void *macOut)
API_AVAILABLE(macos(10.7), ios(6.0));


typedef struct CCCmacContext * CCCmacContextPtr;


/*!
    @function   CCAESCmacCreate
    @abstract   Create a CMac context.

    @param      key         The bytes of the AES key.
    @param      keyLength   The length (in bytes) of the AES key.

    @discussion This returns an AES-CMac context to be used with
                CCAESCmacUpdate(), CCAESCmacFinal() and CCAESCmacDestroy().
 */

CCCmacContextPtr
CCAESCmacCreate(const void *key, size_t keyLength)
API_AVAILABLE(macos(10.10), ios(8.0));

/*!
    @function   CCAESCmacUpdate
    @abstract   Process some data.

    @param      ctx         An HMAC context.
    @param      data        Data to process.
    @param      dataLength  Length of data to process, in bytes.

    @discussion This can be called multiple times.
 */

void CCAESCmacUpdate(CCCmacContextPtr ctx, const void *data, size_t dataLength)
API_AVAILABLE(macos(10.10), ios(8.0));


/*!
    @function   CCAESCmacFinal
    @abstract   Obtain the final Message Authentication Code.

    @param      ctx         A CMAC context.
    @param      macOut      Destination of MAC; allocated by caller.

    @discussion The length of the MAC written to *macOut is 16
         The MAC must be verified by comparing the computed and expected values
         using timingsafe_bcmp. Other comparison functions (e.g. memcmp)
         must not be used as they may be vulnerable to practical timing attacks,
         leading to MAC forgery.
*/

void CCAESCmacFinal(CCCmacContextPtr ctx, void *macOut)
API_AVAILABLE(macos(10.10), ios(8.0));

void
CCAESCmacDestroy(CCCmacContextPtr ctx)
API_AVAILABLE(macos(10.10), ios(8.0));

size_t
CCAESCmacOutputSizeFromContext(CCCmacContextPtr ctx)
API_AVAILABLE(macos(10.10), ios(8.0));


#ifdef __cplusplus
}
#endif

#endif /* _CC_CMACSPI_H_ */
