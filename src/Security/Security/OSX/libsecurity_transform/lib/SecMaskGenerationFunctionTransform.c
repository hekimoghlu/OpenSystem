/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
#include <CoreFoundation/CoreFoundation.h>
#include "SecMaskGenerationFunctionTransform.h"
#include "SecCustomTransform.h"
#include "SecDigestTransform.h"
#include "misc.h"
#include "Utilities.h"

static const CFStringRef kMaskGenerationFunctionTransformName = CFSTR("com.apple.security.MGF1");
static const CFStringRef kLengthName = CFSTR("Length");

static SecTransformInstanceBlock MaskGenerationFunctionTransform(CFStringRef name, 
                                                  SecTransformRef newTransform, 
                                                  SecTransformImplementationRef ref)
{
    __block CFMutableDataRef accumulator = CFDataCreateMutable(NULL, 0);
    __block int32_t outputLength = 0;
    
    SecTransformInstanceBlock instanceBlock = ^{
        SecTransformSetTransformAction(ref, kSecTransformActionFinalize, ^{
            CFReleaseNull(accumulator);
            
            return (CFTypeRef)NULL;
        });
        
        // XXX: be a good citizen, put a validator in for the types.
        
        SecTransformSetAttributeAction(ref, kSecTransformActionAttributeNotification, kLengthName, ^CFTypeRef(SecTransformAttributeRef attribute, CFTypeRef value) {
            CFNumberGetValue((CFNumberRef)value, kCFNumberSInt32Type, &outputLength);
            if (outputLength <= 0) {
                CFErrorRef error = CreateSecTransformErrorRef(kSecTransformErrorInvalidLength, CFSTR("MaskGenerationFunction Length must be one or more (not %@)"), value);
                SecTransformCustomSetAttribute(ref, kSecTransformAbortAttributeName, kSecTransformMetaAttributeValue, error);
                CFSafeRelease(error);
            }

            return (CFTypeRef)NULL;
        });
        
        SecTransformSetAttributeAction(ref, kSecTransformActionAttributeNotification, kSecTransformInputAttributeName, ^CFTypeRef(SecTransformAttributeRef attribute, CFTypeRef value) {
            if (value) {
                CFDataRef d = value;
                CFDataAppendBytes(accumulator, CFDataGetBytePtr(d), CFDataGetLength(d));
            } else {
                int32_t i = 0, l = 0;
                (void)transforms_assume(outputLength > 0);
                CFStringRef digestType = SecTranformCustomGetAttribute(ref, kSecDigestTypeAttribute, kSecTransformMetaAttributeValue);
                SecTransformRef digest0 = transforms_assume(SecDigestTransformCreate(digestType, 0, NULL));
                int32_t digestLength = 0;
                // we've already asserted that digest0 is non-null, but clang doesn't know that
                [[clang::suppress]] {
                    CFNumberRef digestLengthAsCFNumber = SecTransformGetAttribute(digest0, kSecDigestLengthAttribute);
                    CFNumberGetValue(transforms_assume(digestLengthAsCFNumber), kCFNumberSInt32Type, &digestLength);
                }
                (void)transforms_assume(digestLength >= 0);

                UInt8 *buffer = malloc(outputLength + digestLength);
                if (!buffer) {
                    SecTransformCustomSetAttribute(ref, kSecTransformAbortAttributeName, kSecTransformMetaAttributeValue, GetNoMemoryErrorAndRetain());
                    return (CFErrorRef)NULL;
                }

                dispatch_group_t all_hashed = dispatch_group_create();
                dispatch_group_enter(all_hashed);
                for(; l < outputLength; l += digestLength, i++) {
                    dispatch_group_enter(all_hashed);
                    CFErrorRef err = NULL;
                    SecTransformRef digest = NULL;
                    if (l == 0) {
                        digest = digest0;
                    } else {
                        digest = SecDigestTransformCreate(digestType, 0, &err);
                    }
                    
                    if (digest == NULL) {
                        SecTransformCustomSetAttribute(ref, kSecTransformAbortAttributeName, kSecTransformMetaAttributeValue, err);
                        free(buffer);
                        return (CFErrorRef)NULL;
                    }

                    // NOTE: we shuld be able to do this without the copy, make a transform that takes an
                    // array and outputs each item in the array followed by a NULL ought to be quicker.
                    CFMutableDataRef accumulatorPlusCounter = CFDataCreateMutableCopy(NULL, CFDataGetLength(accumulator) + sizeof(uint32_t), accumulator);
                    int32_t bigendian_i = htonl(i);
                    CFDataAppendBytes(accumulatorPlusCounter, (UInt8*)&bigendian_i, sizeof(bigendian_i));
                    SecTransformSetAttribute(digest, kSecTransformInputAttributeName, accumulatorPlusCounter, &err);
                    CFReleaseNull(accumulatorPlusCounter);
                    
                    UInt8 *buf = buffer + l;
                    SecTransformExecuteAsync(digest, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(CFTypeRef message, CFErrorRef error, Boolean isFinal) {
                        if (message) {
                            CFIndex messageLen = CFDataGetLength(message);
                            CFDataGetBytes(message, CFRangeMake(0, messageLen), buf);
                        }
                        if (error) {
                            SecTransformCustomSetAttribute(ref, kSecTransformAbortAttributeName, kSecTransformMetaAttributeValue, error);
                        }
                        if (isFinal) {
                            dispatch_group_leave(all_hashed);
                        }
                    });
                    CFReleaseNull(digest);
                }
                
                dispatch_group_leave(all_hashed);
                dispatch_group_wait(all_hashed, DISPATCH_TIME_FOREVER);
                CFDataRef out = CFDataCreateWithBytesNoCopy(NULL, buffer, outputLength, kCFAllocatorMalloc);
                SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, kSecTransformMetaAttributeValue, out);
                CFReleaseNull(out);
                SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, kSecTransformMetaAttributeValue, NULL);
            }
            return (CFErrorRef)NULL;
        });
        
        return (CFErrorRef)NULL;
    };
    
    return Block_copy(instanceBlock);
}

SecTransformRef SecCreateMaskGenerationFunctionTransform(CFStringRef hashType, int length, CFErrorRef *error)
{
    static dispatch_once_t once;
	__block Boolean ok = TRUE;
    
    if (length <= 0) {
        if (error) {
            *error = CreateSecTransformErrorRef(kSecTransformErrorInvalidLength, CFSTR("MaskGenerationFunction Length must be one or more (not %d)"), length);
        }
        return NULL;
    }
    
    dispatch_once(&once, ^(void) {
        ok = SecTransformRegister(kMaskGenerationFunctionTransformName, MaskGenerationFunctionTransform, error);
    });
        
    if (!ok) {
        return NULL;
    }
    
    SecTransformRef ret = SecTransformCreate(kMaskGenerationFunctionTransformName, error);
    if (!ret) {
        return NULL;
    }
    
    if (!SecTransformSetAttribute(ret, kSecDigestTypeAttribute, hashType ? hashType : kSecDigestSHA1, error)) {
        CFReleaseNull(ret);
        return NULL;
    }

    CFNumberRef len = CFNumberCreate(NULL, kCFNumberIntType, &length);
    ok = SecTransformSetAttribute(ret, kLengthName, len, error);
    CFReleaseNull(len);
    if (!ok) {
        CFReleaseNull(ret);
        return NULL;
    }

    return ret;
}
