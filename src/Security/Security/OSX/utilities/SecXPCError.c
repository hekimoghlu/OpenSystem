/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#include <stdio.h>

#include <utilities/SecXPCError.h>
#include <utilities/SecCFError.h>
#include <utilities/SecCFWrappers.h>
#include <utilities/der_plist.h>

CFStringRef sSecXPCErrorDomain = CFSTR("com.apple.security.xpc");

static const char* kDomainKey = "domain";
static const char* kCodeKey = "code";
static const char* kUserInfoKey = "userinfo";

CFErrorRef SecCreateCFErrorWithXPCObject(xpc_object_t xpc_error)
{
    CFErrorRef result = NULL;

    if (xpc_get_type(xpc_error) == XPC_TYPE_DICTIONARY) {
        CFStringRef domain = NULL;

        const char * domain_string = xpc_dictionary_get_string(xpc_error, kDomainKey);
        if (domain_string != NULL) {
            domain = CFStringCreateWithCString(kCFAllocatorDefault, domain_string, kCFStringEncodingUTF8);
        } else {
            domain = sSecXPCErrorDomain;
            CFRetain(domain);
        }
        CFIndex code = (CFIndex) xpc_dictionary_get_int64(xpc_error, kCodeKey);

        CFTypeRef user_info = NULL;
        size_t size = 0;
        const uint8_t *der = xpc_dictionary_get_data(xpc_error, kUserInfoKey, &size);
        if (der) {
            const uint8_t *der_end = der + size;
            der = der_decode_plist(kCFAllocatorDefault,
                                   &user_info, NULL, der, der_end);
            if (der != der_end)
                CFReleaseNull(user_info);
        }

        result = CFErrorCreate(NULL, domain, code, user_info);

        CFReleaseSafe(user_info);
        CFReleaseSafe(domain);
    } else {
        SecCFCreateErrorWithFormat(kSecXPCErrorUnexpectedType, sSecXPCErrorDomain, NULL, &result, NULL, CFSTR("Remote error not dictionary!: %@"), xpc_error);
    }
    return result;
}

static void SecXPCDictionarySetCFString(xpc_object_t dict, const char *key, CFStringRef string)
{
    CFStringPerformWithCString(string, ^(const char *utf8Str) {
        xpc_dictionary_set_string(dict, key, utf8Str);
    });
}

xpc_object_t SecCreateXPCObjectWithCFError(CFErrorRef error)
{
    xpc_object_t error_xpc = xpc_dictionary_create(NULL, NULL, 0);

    SecXPCDictionarySetCFString(error_xpc, kDomainKey, CFErrorGetDomain(error));
    xpc_dictionary_set_int64(error_xpc, kCodeKey, CFErrorGetCode(error));

    CFDictionaryRef user_info = CFErrorCopyUserInfo(error);
    size_t size = der_sizeof_plist(user_info, NULL);
    if (size) {
        uint8_t *der = malloc(size);
        uint8_t *der_end = der + size;
        uint8_t *der_start = der_encode_plist(user_info, NULL, der, der_end);
        if (der_start) {
            assert(der == der_start);
            xpc_dictionary_set_data(error_xpc, kUserInfoKey, der_start, der_end - der_start);
        }
        free(der);
    }
    CFRelease(user_info);

    return error_xpc;
}
