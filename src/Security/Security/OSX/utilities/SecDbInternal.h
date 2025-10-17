/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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

#ifndef SecDbInternal_h
#define SecDbInternal_h

#include "SecDb.h"

static const size_t kSecDbMaxReaders = 5;

// Do not increase this without changing lock types in SecDb
static const size_t kSecDbMaxWriters = 1;

// maxreaders + maxwriters
static const size_t kSecDbMaxIdleHandles = 6;

// Trustd's databases pass in this constant instead in order
// to reduce trustd's inactive memory footprint by having
// fewer cached open sqlite connections.
static const size_t kSecDbTrustdMaxIdleHandles = 2;

#endif /* SecDbInternal_h */
