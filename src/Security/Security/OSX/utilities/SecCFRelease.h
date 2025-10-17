/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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
#ifndef _SECCFRELEASE_H_
#define _SECCFRELEASE_H_

#include <CoreFoundation/CFBase.h>

// Retains its argument unless it's NULL.  Always returns its argument.
#define CFRetainSafe(CF) ({ __typeof__(CF) _cf = (CF); _cf ? (CFRetain(_cf), _cf) : _cf; })

// Releases CF unless it's NULL.  Always returns NULL, for your convenience and constructs like:
// return CFReleaseSafe(foo);
#define CFReleaseSafe(CF) ({ __typeof__(CF) _cf = (CF); (_cf ? (CFRelease(_cf), ((__typeof__(CF))0)) : _cf); })


#if 0
// Objective-C defines &<bridged-object> to be a constant reference, which we can't assign to, so we
// suffer multiple evaluations of arguments (avoid using it with side-effects) to make things compile well

#define CFTransferRetained(VAR1, VAR2) ({ CFReleaseSafe(VAR1); VAR1 = VAR2; VAR2 = ((__typeof__(VAR2))0)); })

#define CFAssignRetained(VAR, CF) ({ CFReleaseSafe(VAR); VAR1 = CF; CF = ((__typeof__(CF))0)); })

#define CFRetainAssign(VAR,CF) ({ CFReleaseSafe(VAR); VAR = CFRetainSafe(CF); })

#define CFReleaseNull(CF) ({ CFReleaseSafe(CF); CF = ((__typeof__(CF))0)); })

#else

// Assume VAR1 is NULL or an already retained object and CFReleaseSafe VAR1 and assigns VAR2 to VAR1. Returns VAR2.
#define CFTransferRetained(VAR1, VAR2) ({ \
    __typeof__(VAR1) *const _pvar1 = &(VAR1); \
    __typeof__(VAR2) *const _pvar2 = &(VAR2); \
    __typeof__(VAR2) _var2 = *_pvar2; \
    (*_pvar2) = NULL; \
    (*_pvar1) = *_pvar1 ? (CFRelease(*_pvar1), _var2) : _var2; \
})


// Assume CF is NULL or an already retained object and CFReleaseSafe VAR and assigns CF to VAR. Returns CF.
#define CFAssignRetained(VAR,CF) ({ \
    __typeof__(VAR) *const _pvar = &(VAR); \
    __typeof__(CF) _cf = (CF); \
    (*_pvar) = *_pvar ? (CFRelease(*_pvar), _cf) : _cf; \
})

// CFRetainSafe CF and CFReleaseSafe VAR and assigns CF to VAR. Returns CF.
#define CFRetainAssign(VAR,CF) ({ \
    __typeof__(VAR) *const _pvar = &(VAR); \
    __typeof__(CF) _cf = (CF); \
    (((*_pvar) == _cf) ? _cf \
    : ((*_pvar) = (_cf ? (CFRetain(_cf), (*_pvar ? (CFRelease(*_pvar), _cf) : _cf)) \
                   : (CFRelease(*_pvar), ((__typeof__(_cf))0))))); \
})


// Assigns NULL to CF. Releases the value stored at CF unless it was NULL.  Always returns NULL, for your convenience
#define CFReleaseNull(CF) ({ __typeof__(CF) *const _pcf = &(CF), _cf = *_pcf; (_cf ? (*_pcf) = ((__typeof__(CF))0), (CFRelease(_cf), ((__typeof__(CF))0)) : _cf); })

#endif

#endif /* _SECCFRELEASE_H_ */
