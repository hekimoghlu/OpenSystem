/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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
#include <cstdio>

struct MoveOnly {
    int id;
    MoveOnly() : id(0) { printf("MoveOnly %d created\n", id); }
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly(MoveOnly&& other) : id(other.id + 1) { printf("MoveOnly %d move-created\n", id); }
    ~MoveOnly() { printf("MoveOnly %d destroyed\n", id); }
};

struct Copyable {
    int id;
    Copyable() : id(0) { printf("Copyable %d created\n", id); }
    Copyable(const Copyable& other) : id(other.id + 1) { printf("Copyable %d copy-created\n", id); }
    Copyable(Copyable&& other) : id(other.id + 1) { printf("Copyable %d move-created\n", id); }
    ~Copyable() { printf("Copyable %d destroyed\n", id); }
};

inline void byRValueRef(MoveOnly&& x) {}
inline void byRValueRef(Copyable&& x) {}
