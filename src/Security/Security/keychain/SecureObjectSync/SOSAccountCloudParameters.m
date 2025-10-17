/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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

//
//  AccountCloudParameters.c
//  sec
//

#include "SOSAccountPriv.h"
#include "keychain/SecureObjectSync/SOSTransportKeyParameter.h"
#include "keychain/SecureObjectSync/SOSCircleDer.h"
//
// Cloud Paramters encode/decode
//

static size_t der_sizeof_cloud_parameters(SecKeyRef publicKey, CFDataRef paramters, CFErrorRef* error)
{
    size_t public_key_size = der_sizeof_public_bytes(publicKey, error);
    size_t parameters_size = der_sizeof_data_or_null(paramters, error);
    
    return ccder_sizeof(CCDER_CONSTRUCTED_SEQUENCE, public_key_size + parameters_size);
}

static uint8_t* der_encode_cloud_parameters(SecKeyRef publicKey, CFDataRef paramters, CFErrorRef* error,
                                            const uint8_t* der, uint8_t* der_end)
{
    uint8_t* original_der_end = der_end;
    
    return ccder_encode_constructed_tl(CCDER_CONSTRUCTED_SEQUENCE, original_der_end, der,
                                       der_encode_public_bytes(publicKey, error, der,
                                                               der_encode_data_or_null(paramters, error, der, der_end)));
}

const uint8_t* der_decode_cloud_parameters(CFAllocatorRef allocator,
                                                  CFIndex algorithmID, SecKeyRef* publicKey,
                                                  CFDataRef *pbkdfParams,
                                                  CFErrorRef* error,
                                                  const uint8_t* der, const uint8_t* der_end)
{
    const uint8_t *sequence_end;
    der = ccder_decode_sequence_tl(&sequence_end, der, der_end);
    der = der_decode_public_bytes(allocator, algorithmID, publicKey, error, der, sequence_end);
    der = der_decode_data_or_null(allocator, pbkdfParams, error, der, sequence_end);
    
    return der;
}


bool SOSAccountPublishCloudParameters(SOSAccount* account, CFErrorRef* error){
    bool success = false;
    CFIndex cloud_der_len = der_sizeof_cloud_parameters(account.accountKey,
                                                        (__bridge CFDataRef)(account.accountKeyDerivationParameters),
                                                        error);

    CFMutableDataRef cloudParameters = CFDataCreateMutableWithScratch(kCFAllocatorDefault, cloud_der_len);
    
    if (der_encode_cloud_parameters(account.accountKey, (__bridge CFDataRef)(account.accountKeyDerivationParameters), error,
                                    CFDataGetMutableBytePtr(cloudParameters),
                                    CFDataGetMutablePastEndPtr(cloudParameters)) != NULL) {

        CFErrorRef changeError = NULL;

        if ([account.key_transport SOSTransportKeyParameterPublishCloudParameters:account.key_transport data:cloudParameters err:error]) {
            success = true;
        } else {
            SOSCreateErrorWithFormat(kSOSErrorSendFailure, changeError, error, NULL,
                                     CFSTR("update parameters key failed [%@]"), cloudParameters);
        }
        CFReleaseSafe(changeError);
    } else {
        SOSCreateError(kSOSErrorEncodeFailure, CFSTR("Encoding parameters failed"), NULL, error);
    }
    
    CFReleaseNull(cloudParameters);
    
    return success;
}

bool SOSAccountRetrieveCloudParameters(SOSAccount* account, SecKeyRef *newKey,
                                       CFDataRef derparms,
                                       CFDataRef *pbkdfParams, CFErrorRef* error) {
    const uint8_t *parse_end = der_decode_cloud_parameters(kCFAllocatorDefault, kSecECDSAAlgorithmID,
                                                           newKey, pbkdfParams, error,
                                                           CFDataGetBytePtr(derparms), CFDataGetPastEndPtr(derparms));
    
    if (parse_end == CFDataGetPastEndPtr(derparms)) return true;
    return false;
}

