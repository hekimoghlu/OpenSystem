/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Trim11.h: Trim support utility class.

#ifndef LIBANGLE_RENDERER_D3D_D3D11_TRIM11_H_
#define LIBANGLE_RENDERER_D3D_D3D11_TRIM11_H_

#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/angletypes.h"

#if defined(ANGLE_ENABLE_WINDOWS_UWP)
#    include <EventToken.h>
#endif

namespace rx
{
class Renderer11;

class Trim11 : angle::NonCopyable
{
  public:
    explicit Trim11(Renderer11 *renderer);
    ~Trim11();

  private:
    Renderer11 *mRenderer;
#if defined(ANGLE_ENABLE_WINDOWS_UWP)
    EventRegistrationToken mApplicationSuspendedEventToken;
#endif

    void trim();
    bool registerForRendererTrimRequest();
    void unregisterForRendererTrimRequest();
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_TRIM11_H_
