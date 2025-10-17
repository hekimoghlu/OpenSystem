/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_PRAGMA_H_
#define COMPILER_TRANSLATOR_PRAGMA_H_

struct TPragma
{
    struct STDGL
    {
        STDGL() : invariantAll(false) {}

        bool invariantAll;
    };

    // By default optimization is turned on and debug is turned off.
    TPragma() : optimize(true), debug(false) {}
    TPragma(bool o, bool d) : optimize(o), debug(d) {}

    bool optimize;
    bool debug;
    STDGL stdgl;
};

#endif  // COMPILER_TRANSLATOR_PRAGMA_H_
