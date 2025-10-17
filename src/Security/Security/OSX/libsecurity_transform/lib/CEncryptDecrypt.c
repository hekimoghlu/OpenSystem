/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#include <CoreFoundation/CoreFoundation.h>
#include "corecrypto/ccsha1.h"
#include "corecrypto/ccrsa_priv.h"
#include "corecrypto/ccrng.h"
#include "corecrypto/ccn.h"
#include "stdio.h"
#include "misc.h"
#include "Utilities.h"

// corecrypto headers don't like C++ (on a deaper level then extern "C" {} can fix
// so we need a C "shim" for all our corecrypto use.

CFDataRef oaep_padding_via_c(int desired_message_length, CFDataRef dataValue);
CFDataRef oaep_padding_via_c(int desired_message_length, CFDataRef dataValue) CF_RETURNS_RETAINED
{
    size_t pBufferSize = ccn_sizeof_size(desired_message_length);
	cc_unit *paddingBuffer = malloc(pBufferSize);
    if (paddingBuffer == NULL){
        return (void*)GetNoMemoryErrorAndRetain();
    }
    
	bzero(paddingBuffer, pBufferSize); // XXX needed??

    ccrsa_oaep_encode(ccsha1_di(),
                      ccrng(NULL),
                      pBufferSize, (cc_unit*)paddingBuffer,
                      CFDataGetLength(dataValue), CFDataGetBytePtr(dataValue));
    ccn_swap(ccn_nof_size(pBufferSize), (cc_unit*)paddingBuffer);
	
    CFDataRef paddedValue = CFDataCreate(NULL, (UInt8*)paddingBuffer, desired_message_length);
    free(paddingBuffer);
    return paddedValue ? paddedValue : (void*)GetNoMemoryErrorAndRetain();
}

CFDataRef oaep_unpadding_via_c(CFDataRef encodedMessage);
CFDataRef oaep_unpadding_via_c(CFDataRef encodedMessage) CF_RETURNS_RETAINED
{
	size_t mlen = CFDataGetLength(encodedMessage);
	size_t pBufferSize = ccn_sizeof_size(mlen);
    cc_unit *paddingBuffer = malloc(pBufferSize);
	UInt8 *plainText = malloc(mlen);
    if (plainText == NULL || paddingBuffer == NULL) {
        free(plainText);
        free(paddingBuffer);
        return (void*)GetNoMemoryErrorAndRetain();
    }
	
	ccn_read_uint(ccn_nof_size(mlen), paddingBuffer, mlen, CFDataGetBytePtr(encodedMessage));
	size_t plainTextLength = mlen;
    int err = ccrsa_oaep_decode(ccsha1_di(), &plainTextLength, plainText, mlen, paddingBuffer);
	
    if (err) {
		// XXX should make a CFError or something.
        CFErrorRef error = fancy_error(CFSTR("CoreCrypto"), err, CFSTR("OAEP decode error"));
        CFRetainSafe(error);
        free(plainText);
        free(paddingBuffer);
        return (void*)error;
    }
	
    CFDataRef result = CFDataCreate(NULL, (UInt8*)plainText, plainTextLength);
    
    free(plainText);
    free(paddingBuffer);
    
    return result;
}
