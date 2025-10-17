/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
#ifndef XNU_LIBKERN_LIBKERN_CXX_OS_BOUNDED_ARRAY_REF_H
#define XNU_LIBKERN_LIBKERN_CXX_OS_BOUNDED_ARRAY_REF_H

#if !TAPI

#if DRIVERKIT_FRAMEWORK_INCLUDE
#include <DriverKit/bounded_array_ref.h>
#include <DriverKit/OSBoundedPtr.h>
#else
#include <libkern/c++/bounded_array_ref.h>
#include <libkern/c++/OSBoundedPtr.h>
#endif /* DRIVERKIT_FRAMEWORK_INCLUDE */


template <typename T>
using OSBoundedArrayRef = libkern::bounded_array_ref<T, os_detail::panic_trapping_policy>;

#endif /* !TAPI */

#endif /* !XNU_LIBKERN_LIBKERN_CXX_OS_BOUNDED_ARRAY_REF_H */
