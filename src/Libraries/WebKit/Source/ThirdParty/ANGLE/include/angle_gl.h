/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
// angle_gl.h:
//   Includes all necessary GL headers and definitions for ANGLE.
//

#ifndef ANGLEGL_H_
#define ANGLEGL_H_

#include "GLES/gl.h"
#include "GLES/glext.h"
#include "GLES2/gl2.h"
#include "GLES2/gl2ext.h"
#include "GLES3/gl3.h"
#include "GLES3/gl31.h"
#include "GLES3/gl32.h"

// TODO(http://anglebug.com/42262388): Autogenerate these enums from gl.xml
// HACK: Defines for queries that are not in GLES
#define GL_CONTEXT_PROFILE_MASK 0x9126
#define GL_CONTEXT_COMPATIBILITY_PROFILE_BIT 0x00000002
#define GL_CONTEXT_CORE_PROFILE_BIT 0x00000001

#endif  // ANGLEGL_H_
