/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
/***********************************************************************
* threading.h
* Threading support
**********************************************************************/

#ifndef _OBJC_THREADING_H
#define _OBJC_THREADING_H

// TLS key identifiers
enum class tls_key {
    main                       = 0,
    sync_data                  = 1,
    sync_count                 = 2,
    autorelease_pool           = 3,
#if SUPPORT_RETURN_AUTORELEASE
    return_autorelease_object  = 4,
    return_autorelease_address = 5
#endif
};

#if OBJC_THREADING_PACKAGE == OBJC_THREADING_NONE
#include "nothreads.h"
#elif OBJC_THREADING_PACKAGE == OBJC_THREADING_DARWIN
#include "darwin.h"
#elif OBJC_THREADING_PACKAGE == OBJC_THREADING_PTHREADS
#include "pthreads.h"
#elif OBJC_THREADING_PACKAGE == OBJC_THREADING_C11THREADS
#include "c11threads.h"
#else
#error No threading package selected in objc-config.h
#endif

#include "mixins.h"
#include "lockdebug.h"
#include "tls.h"

using objc_lock_t = locker_mixin<lockdebug::lock_mixin<objc_lock_base_t>>;
using objc_recursive_lock_t =
    locker_mixin<lockdebug::lock_mixin<objc_recursive_lock_base_t>>;
using objc_nodebug_lock_t = locker_mixin<objc_lock_base_t>;

#endif // _OBJC_THREADING_H
