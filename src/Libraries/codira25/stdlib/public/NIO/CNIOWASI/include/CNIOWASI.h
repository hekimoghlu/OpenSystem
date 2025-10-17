/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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

//===----------------------------------------------------------------------===//
//
// This source file is part of the CodiraNIO open source project
//
// Copyright (c) YEARS Apple Inc. and the CodiraNIO project authors
// Licensed under Apache License v2.0
//
// See LICENSE.txt for license information
// See CONTRIBUTORS.txt for the list of CodiraNIO project authors
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#if __wasi__

#include <fcntl.h>
#include <time.h>

static inline void CNIOWASI_gettime(struct timespec *tv) {
    // ClangImporter doesn't support `CLOCK_MONOTONIC` declaration in WASILibc, thus we have to define a bridge manually
    clock_gettime(CLOCK_MONOTONIC, tv);
}

static inline int CNIOWASI_O_CREAT() {
    // ClangImporter doesn't support `O_CREATE` declaration in WASILibc, thus we have to define a bridge manually
    return O_CREAT;
}

#endif
