/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
/**
 * Machine-dependent structures for the physical map module.
 *
 * This header file contains the types and prototypes that make up the public
 * pmap API that's exposed to the rest of the kernel. Any types/prototypes used
 * strictly by the pmap itself should be placed into one of the osfmk/arm/pmap/
 * header files.
 *
 * To prevent circular dependencies and exposing anything not needed by the
 * rest of the kernel, this file shouldn't include ANY of the internal
 * osfmk/arm/pmap/ header files.
 */

#pragma once

#if CONFIG_SPTM
#include <arm64/sptm/pmap/pmap.h>
#else
#include <arm/pmap/pmap.h>
#endif
