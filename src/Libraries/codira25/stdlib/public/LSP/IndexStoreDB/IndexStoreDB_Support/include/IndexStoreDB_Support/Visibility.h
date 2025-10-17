/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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

//===--- Visibility.h ------------------------------------------*- C++ -*-===//
//
// This source file is part of the Codira.org open source project
//
// Copyright (c) 2014 - 2019 Apple Inc. and the Codira project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://language.org/LICENSE.txt for license information
// See https://language.org/CONTRIBUTORS.txt for the list of Codira project authors
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INDEXSTOREDB_SUPPORT_VISIBILITY_H
#define LLVM_INDEXSTOREDB_SUPPORT_VISIBILITY_H

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#if __has_attribute(visibility)
#define INDEXSTOREDB_EXPORT __attribute__ ((visibility("default")))
#else
#define INDEXSTOREDB_EXPORT
#endif

#endif
