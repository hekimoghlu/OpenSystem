/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
#ifndef CommonRSACryptorSPI_h
#define CommonRSACryptorSPI_h

#include <CommonCrypto/CommonRSACryptor.h>

#ifdef __cplusplus
extern "C" {
#endif
    


// The following SPIs return the RSA CRT parameters
    
/*!
@function   CCRSAGetCRTComponentsSizes
@abstract   Returns the size of the RSA CRT components dp, dq, qinv, where dp = d (mod p) and dq = d (mod q) and qinv = q ^ -1 (mod p)
@param      rsaKey     A pointer to a CCRSACryptorRef.
@param      dpSize     Input pointer to return the size of the RSA CRT parameter dp.
 param      dqSize     Input pointer to return the size of the RSA CRT parameter dq.
@param      qinvSize   Input pointer to return the size of the RSA CRT parameter qinv.
@result                If the function is successful (kCCSuccess)
*/

CCCryptorStatus CCRSAGetCRTComponentsSizes(CCRSACryptorRef rsaKey, size_t *dpSize, size_t *dqSize, size_t *qinvSize);

/*!
 @function   CCRSAGetCRTComponents
 @abstract   Returns the RSA CRT components dp, dq and qinv in big endian form. The required size of dp, dq and qinv buffers can be obtained by calling to CCRSAGetCRTDpSize, CCRSAGetCRTDqSize and CCRSAGetCRTQinvSize SPIs, respectively
 @param      rsaKey     A pointer to a CCRSACryptorRef
 @param      dp         The buffer to return the RSA CRT parameter dp
 @param      dpSize     The size of the input buffer dp
 @param      dq       	The buffer to return the RSA CRT parameter dq
 @param      dqSize     The size of the input buffer dq
 @param      qinv       The buffer to return the RSA CRT parameter qinv
 @param      qinvSize   The size of the input buffer qinv
 @result                If the function is successful (kCCSuccess)
 */
CCCryptorStatus CCRSAGetCRTComponents(CCRSACryptorRef rsaKey, void *dp, size_t dpSize, void *dq, size_t dqSize, void *qinv, size_t qinvSize);


#ifdef __cplusplus
}
#endif


#endif /* CommonRSACryptorSPI_h */
