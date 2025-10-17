/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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
#ifndef libsecurity_transform_SecMaskGenerationFunctionTransform_h
#define libsecurity_transform_SecMaskGenerationFunctionTransform_h

#include "SecTransform.h"

#ifdef __cplusplus
extern "C" {
#endif
    
/*!
 @function SecMaskGenerationFunctionTransformCreate
 @abstract			Creates a MGF computation object.
 @param hashType	The type of digest to compute the MGF with.  You may pass NULL
 for this parameter, in which case an appropriate
 algorithm will be chosen for you (SHA1 for MGF1).
 @param length	The desired digest length.  Note that certain
 algorithms may only support certain sizes. You may
 pass 0 for this parameter, in which case an
 appropriate length will be chosen for you.
 @param error		A pointer to a CFErrorRef.  This pointer will be set
 if an error occurred.  This value may be NULL if you
 do not want an error returned.
 @result				A pointer to a SecTransformRef object.  This object must
 be released with CFRelease when you are done with
 it.  This function will return NULL if an error
 occurred.
 @discussion			This function creates a transform which computes a
 fixed length (maskLength) deterministic pseudorandom output.
 */
    
    
SecTransformRef SecCreateMaskGenerationFunctionTransform(CFStringRef hashType, int length, CFErrorRef *error)
API_DEPRECATED("SecTransform is no longer supported", macos(10.7, 13.0)) API_UNAVAILABLE(ios, tvos, watchos, macCatalyst);
    
#ifdef __cplusplus
}
#endif

#endif
