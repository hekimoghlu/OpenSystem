/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
//  trustd_objc_helpers.h
//  Security
//
//

#ifndef trustd_objc_helpers_h
#define trustd_objc_helpers_h

static inline bool isNSString(id nsType) {
    return nsType && [nsType isKindOfClass:[NSString class]];
}

static inline bool isNSNumber(id nsType) {
    return nsType && [nsType isKindOfClass:[NSNumber class]];
}

static inline bool isNSDate(id nsType) {
    return nsType && [nsType isKindOfClass:[NSDate class]];
}

static inline bool isNSData(id nsType) {
    return nsType && [nsType isKindOfClass:[NSData class]];
}

static inline bool isNSArray(id nsType) {
    return nsType && [nsType isKindOfClass:[NSArray class]];
}

static inline bool isNSDictionary(id nsType) {
    return nsType && [nsType isKindOfClass:[NSDictionary class]];
}

#endif /* trustd_objc_helpers_h */
