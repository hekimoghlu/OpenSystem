/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// NativeWindow9.cpp: Defines NativeWindow9, a class for managing and
// performing operations on an EGLNativeWindowType for the D3D9 renderer.

#include "libANGLE/renderer/d3d/d3d9/NativeWindow9.h"

namespace rx
{
NativeWindow9::NativeWindow9(EGLNativeWindowType window) : NativeWindowD3D(window) {}

bool NativeWindow9::initialize()
{
    return true;
}

bool NativeWindow9::getClientRect(LPRECT rect) const
{
    return GetClientRect(getNativeWindow(), rect) == TRUE;
}

bool NativeWindow9::isIconic() const
{
    return IsIconic(getNativeWindow()) == TRUE;
}

// static
bool NativeWindow9::IsValidNativeWindow(EGLNativeWindowType window)
{
    return IsWindow(window) == TRUE;
}

}  // namespace rx
