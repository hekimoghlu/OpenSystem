/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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

#include <Foundation/Foundation.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-extension"
#pragma clang assume_nonnull begin

struct CxxRefType {
} __attribute__((language_attr("import_reference")))
__attribute__((language_attr("retain:retainCxxRefType")))
__attribute__((language_attr("release:releaseCxxRefType")));

struct CxxValType {};

void retainCxxRefType(CxxRefType *_Nonnull b) {}
void releaseCxxRefType(CxxRefType *_Nonnull b) {}

@interface Bridge : NSObject

+ (struct CxxRefType *)objCMethodReturningFRTUnannotated;
+ (struct CxxRefType *)objCMethodReturningFRTUnowned
    __attribute__((language_attr("returns_unretained")));
+ (struct CxxRefType *)objCMethodReturningFRTOwned
    __attribute__((language_attr("returns_retained")));
+ (struct CxxRefType *)objCMethodReturningFRTBothAnnotations // expected-error {{'objCMethodReturningFRTBothAnnotations' cannot be annotated with both LANGUAGE_RETURNS_RETAINED and LANGUAGE_RETURNS_UNRETAINED}}
    __attribute__((language_attr("returns_unretained")))
    __attribute__((language_attr("returns_retained")));
+ (struct CxxValType *)objCMethodReturningNonCxxFrtAnannotated // expected-error {{'objCMethodReturningNonCxxFrtAnannotated' cannot be annotated with either LANGUAGE_RETURNS_RETAINED or LANGUAGE_RETURNS_UNRETAINED because it is not returning a LANGUAGE_SHARED_REFERENCE type}}
    __attribute__((language_attr("returns_retained")));

@end

@implementation Bridge
+ (struct CxxRefType *)objCMethodReturningFRTUnannotated {
}
+ (struct CxxRefType *)objCMethodReturningFRTUnowned
    __attribute__((language_attr("returns_unretained"))) {
}
+ (struct CxxRefType *)objCMethodReturningFRTOwned
    __attribute__((language_attr("returns_retained"))) {
}
+ (struct CxxRefType *)objCMethodReturningFRTBothAnnotations
    __attribute__((language_attr("returns_unretained")))
    __attribute__((language_attr("returns_retained"))) {
}
+ (struct CxxValType *)objCMethodReturningNonCxxFrtAnannotated
    __attribute__((language_attr("returns_retained"))) {
}

@end

#pragma clang diagnostic pop
#pragma clang assume_nonnull end
