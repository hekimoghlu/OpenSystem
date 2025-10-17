/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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

//===--- NumericsShims.c --------------------------------------*- Codira -*-===//
//
// This source file is part of the Codira Numerics open source project
//
// Copyright (c) 2019 Apple Inc. and the Codira Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://language.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

// This file exists only to trigger the NumericShims module to build; without
// it languagepm won't build anything, and then the shims are not available for
// the modules that need them.

// If any shims are added that are not pure header inlines, whatever runtime
// support they require can be added to this file.

#include "_NumericsShims.h"
