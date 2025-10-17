/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
#include "config.h"
#include "RefPtrUdev.h"

#include <libudev.h>

namespace WTF {

struct udev* DefaultRefDerefTraits<struct udev>::refIfNotNull(struct udev* ptr)
{
    if (LIKELY(ptr))
        udev_ref(ptr);
    return ptr;
}

void DefaultRefDerefTraits<struct udev>::derefIfNotNull(struct udev* ptr)
{
    if (LIKELY(ptr))
        udev_unref(ptr);
}

struct udev_device* DefaultRefDerefTraits<struct udev_device>::refIfNotNull(struct udev_device* ptr)
{
    if (LIKELY(ptr))
        udev_device_ref(ptr);
    return ptr;
}

void DefaultRefDerefTraits<struct udev_device>::derefIfNotNull(struct udev_device* ptr)
{
    if (LIKELY(ptr))
        udev_device_unref(ptr);
}

struct udev_enumerate* DefaultRefDerefTraits<struct udev_enumerate>::refIfNotNull(struct udev_enumerate* ptr)
{
    if (LIKELY(ptr))
        udev_enumerate_ref(ptr);
    return ptr;
}

void DefaultRefDerefTraits<struct udev_enumerate>::derefIfNotNull(struct udev_enumerate* ptr)
{
    if (LIKELY(ptr))
        udev_enumerate_unref(ptr);
}

} // namespace WTF
