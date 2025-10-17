/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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
#ifndef _SECURITY_AUTH_OBJECT_H_
#define _SECURITY_AUTH_OBJECT_H_

#include "authd_private.h"
#include <CoreFoundation/CoreFoundation.h>
#include <CoreFoundation/CFRuntime.h>

#if defined(__cplusplus)
extern "C" {
#endif
    
#define __AUTH_BASE_STRUCT_HEADER__ \
    CFRuntimeBase _base;
    
struct _auth_base_s {
    __AUTH_BASE_STRUCT_HEADER__;
};

#define AUTH_TYPE(type) const CFRuntimeClass type
    
#define AUTH_TYPE_INSTANCE(name, ...) \
    AUTH_TYPE(_auth_type_##name) = { \
        .version = 0, \
        .className = #name "_t", \
        __VA_ARGS__ \
    }

#define AUTH_CLASS_SIZE(name) (sizeof(struct _##name##_s) - sizeof(CFRuntimeBase))
    
#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_AUTH_OBJECT_H_ */
