/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Win32Pixmap.h: Definition of the implementation of OSPixmap for Win32 (Windows)

#ifndef UTIL_WIN32_PIXMAP_H_
#define UTIL_WIN32_PIXMAP_H_

#include <windows.h>

#include "util/OSPixmap.h"

class Win32Pixmap : public OSPixmap
{
  public:
    Win32Pixmap();
    ~Win32Pixmap() override;

    bool initialize(EGLNativeDisplayType display, size_t width, size_t height, int depth) override;

    EGLNativePixmapType getNativePixmap() const override;

  private:
    HBITMAP mBitmap;
};

#endif  // UTIL_WIN32_PIXMAP_H_
