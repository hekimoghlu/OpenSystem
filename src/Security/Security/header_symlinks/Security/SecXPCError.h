/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
#ifndef _UTILITIES_SECXPCERROR_H_
#define _UTILITIES_SECXPCERROR_H_

#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFError.h>
#include <xpc/xpc.h>

__BEGIN_DECLS

extern CFStringRef sSecXPCErrorDomain;

enum {
    kSecXPCErrorSuccess = 0,
    kSecXPCErrorUnexpectedType = 1,
    kSecXPCErrorUnexpectedNull = 2,
    kSecXPCErrorConnectionFailed = 3,
    kSecXPCErrorUnknown = 4,
};

CFErrorRef SecCreateCFErrorWithXPCObject(xpc_object_t xpc_error);
xpc_object_t SecCreateXPCObjectWithCFError(CFErrorRef error);

__END_DECLS

#endif /* UTILITIES_SECXPCERROR_H */
