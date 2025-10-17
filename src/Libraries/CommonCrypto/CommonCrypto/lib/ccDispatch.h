/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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
//  ccDispatch.h
//  CommonCrypto
//

#ifndef ccDispatch_h
#define ccDispatch_h

#if defined (_WIN32)
    #include <windows.h>
    #define dispatch_once_t  INIT_ONCE
    typedef void (*dispatch_function_t)(void *);
    void cc_dispatch_once(dispatch_once_t *predicate, void *context, dispatch_function_t function);
#else
    #include <dispatch/dispatch.h>
    #define cc_dispatch_once(predicate, context, function) dispatch_once_f(predicate, context, function)
#endif

#endif /* ccDispatch_h */
