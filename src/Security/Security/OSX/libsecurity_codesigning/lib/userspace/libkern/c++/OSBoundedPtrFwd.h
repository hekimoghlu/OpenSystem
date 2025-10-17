/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 22, 2024.
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
#ifndef XNU_LIBKERN_LIBKERN_CXX_OS_BOUNDED_PTR_FWD_H
#define XNU_LIBKERN_LIBKERN_CXX_OS_BOUNDED_PTR_FWD_H

#if !TAPI

#if DRIVERKIT_FRAMEWORK_INCLUDE
#include <DriverKit/IOLib.h>
#include <DriverKit/bounded_ptr_fwd.h>
#else
#include <kern/debug.h>
#include <libkern/c++/bounded_ptr_fwd.h>
#endif /* DRIVERKIT_FRAMEWORK_INCLUDE */

namespace os_detail {
struct panic_trapping_policy;
}

template <typename T>
using OSBoundedPtr = libkern::bounded_ptr<T, os_detail::panic_trapping_policy>;

#endif /* !TAPI */

#endif /* !XNU_LIBKERN_LIBKERN_CXX_OS_BOUNDED_PTR_FWD_H */
