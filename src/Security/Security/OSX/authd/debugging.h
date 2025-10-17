/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
#ifndef _SECURITY_AUTH_DEBUGGING_H_
#define _SECURITY_AUTH_DEBUGGING_H_

#if defined(__cplusplus)
extern "C" {
#endif

#include <os/log.h>
#include <os/activity.h>

#define AUTHD_DEFINE_LOG \
static os_log_t AUTHD_LOG_DEFAULT(void) { \
static dispatch_once_t once; \
static os_log_t log; \
dispatch_once(&once, ^{ log = os_log_create("com.apple.Authorization", "authd"); }); \
return log; \
};

#define AUTHD_LOG AUTHD_LOG_DEFAULT()

#ifndef CFReleaseSafe
#define CFReleaseSafe(CF) { CFTypeRef _cf = (CF); if (_cf) CFRelease(_cf); }
#endif
#ifndef CFReleaseNull
#define CFReleaseNull(CF) { CFTypeRef _cf = (CF); \
    if (_cf) { (CF) = NULL; CFRelease(_cf); } }
#endif
#ifndef CFRetainSafe
#define CFRetainSafe(CF) { CFTypeRef _cf = (CF); if (_cf) CFRetain(_cf); }
#endif
#define CFAssignRetained(VAR,CF) ({ \
__typeof__(VAR) *const _pvar = &(VAR); \
__typeof__(CF) _cf = (CF); \
(*_pvar) = *_pvar ? (CFRelease(*_pvar), _cf) : _cf; \
})

#define xpc_release_safe(obj)  if (obj) { xpc_release(obj); obj = NULL; }
#define free_safe(obj)  if (obj) { free(obj); obj = NULL; }
    
#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_AUTH_DEBUGGING_H_ */
