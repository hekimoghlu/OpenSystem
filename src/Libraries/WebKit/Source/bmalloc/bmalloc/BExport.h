/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#pragma once

#include "BPlatform.h"

#if BUSE(DECLSPEC_ATTRIBUTE)
#define BEXPORT_DECLARATION __declspec(dllexport)
#define BIMPORT_DECLARATION __declspec(dllimport)
#elif BUSE(VISIBILITY_ATTRIBUTE)
#define BEXPORT_DECLARATION __attribute__((visibility("default")))
#define BIMPORT_DECLARATION BEXPORT_DECLARATION
#else
#define BEXPORT_DECLARATION
#define BIMPORT_DECLARATION
#endif

#if !defined(BEXPORT)

#if defined(BUILDING_bmalloc) || defined(STATICALLY_LINKED_WITH_bmalloc)
#define BEXPORT BEXPORT_DECLARATION
#else
#define BEXPORT BIMPORT_DECLARATION
#endif

#endif

#define BNOEXPORT
