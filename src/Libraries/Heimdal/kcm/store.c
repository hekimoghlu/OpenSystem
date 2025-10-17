/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#include "kcm_locl.h"

#include <Kernel/IOKit/crypto/AppleFDEKeyStoreDefs.h>
#include <IOKit/IOBSD.h>
#include <IOKit/IOKitLib.h>

#include <uuid/uuid.h>
#include <stdio.h>
#include <stdarg.h>

static io_connect_t
openiodev(void)
{
    io_registry_entry_t service;
    CFMutableDictionaryRef matching;
    io_connect_t conn;
    kern_return_t kr;
    
    matching = IOServiceMatching(kAppleFDEKeyStoreServiceName);
    if (matching == NULL)
	return IO_OBJECT_NULL;

    service = IOServiceGetMatchingService(kIOMasterPortDefault, matching);
    if (service == IO_OBJECT_NULL)
	return IO_OBJECT_NULL;
    
    kr = IOServiceOpen(service, mach_task_self(), 0, &conn);
    IOObjectRelease(service);
    if (kr != KERN_SUCCESS)
	return IO_OBJECT_NULL;
    
    kr = IOConnectCallMethod(conn, kAppleFDEKeyStoreUserClientOpen, NULL, 0, NULL, 0, NULL, NULL, NULL, NULL);
    if (kr != KERN_SUCCESS) {
	IOServiceClose(conn);
	return IO_OBJECT_NULL;
    }
    
    return conn;
}

static void
closeiodev(io_connect_t conn)
{
    kern_return_t kr;

    kr = IOConnectCallMethod(conn, kAppleFDEKeyStoreUserClientClose, NULL, 0, NULL, 0, NULL, NULL, NULL, NULL);
    if (kr != KERN_SUCCESS)
	return;

    IOServiceClose(conn);
}

krb5_error_code
kcm_create_key(krb5_uuid uuid)
{
    io_connect_t conn;
    createKeyGetUUID_InStruct_t createKey;
    kern_return_t kr;
    uuid_OutStruct_t key;
    size_t outputStructSize = sizeof(key);

    conn = openiodev();
    if (conn == IO_OBJECT_NULL)
	return EINVAL;
    
    createKey.keySizeInBytes = V1_KEYSIZE;
    createKey.algorithm = fDE_ALG_AESXTS;
    
    memset(&key, 0, sizeof(key));
    
    kr = IOConnectCallMethod(conn, kAppleFDEKeyStore_createKeyGetUUID,
			     NULL, 0,
			     &createKey, sizeof(createKey),
			     NULL, 0,
			     &key, &outputStructSize);
    closeiodev(conn);
    if (kr != KERN_SUCCESS)
	return EINVAL;
    
    memcpy(uuid, key.uuid, sizeof(key.uuid));
    
    return 0;
}

krb5_error_code
kcm_store_io(krb5_context context,
	     krb5_uuid uuid,
	     void *ptr,
	     size_t length,
	     krb5_data *data,
	     bool encrypt)
{
    xtsEncrypt_InStruct_t xtsEncrypt_InStruct;
    size_t inseed_size = 64;
    io_connect_t conn;
    kern_return_t kr;
    uint8_t *inseed;
    krb5_crypto crypto = NULL;
    krb5_error_code ret;
    
    krb5_data_zero(data);

    inseed = malloc(inseed_size);
    if (inseed == NULL)
	err(1, "malloc");

    memset(inseed, 0, inseed_size);
    
    conn = openiodev();
    if (conn == IO_OBJECT_NULL) {
	free(inseed);
	return EINVAL;
    }

    uuid_copy(xtsEncrypt_InStruct.key_uuid, uuid);
    xtsEncrypt_InStruct.bufferAddress = (uint64_t) (intptr_t) inseed;
    xtsEncrypt_InStruct.bufferLength = (uint64_t) inseed_size;
    memset(xtsEncrypt_InStruct.tweak, 0, XTS_TWEAK_BYTES);
    
    kr = IOConnectCallMethod(conn, kAppleFDEKeyStore_xtsEncrypt, 
			     NULL, 0, 
			     & xtsEncrypt_InStruct, sizeof(xtsEncrypt_InStruct), 
			     NULL, 0,
			     NULL, 0);
    closeiodev(conn);
    if (kr != KERN_SUCCESS) {
	free(inseed);
	return EINVAL;
    }
    
    CC_SHA256(inseed, (CC_LONG)inseed_size, inseed);

    krb5_keyblock keyblock;
    keyblock.keytype = ETYPE_AES128_CTS_HMAC_SHA1_96;
    keyblock.keyvalue.data = inseed;
    keyblock.keyvalue.length = 16;
    
    ret = krb5_crypto_init(context, &keyblock, 0, &crypto);
    free(inseed);
    if (ret)
	return ret;

    if (encrypt)
	ret = krb5_encrypt(context, crypto, 1, ptr, length, data);
    else
	ret = krb5_decrypt(context, crypto, 1, ptr, length, data);

    krb5_crypto_destroy(context, crypto);
    
    return ret;
}
