/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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

// We do not delete "delete" operators to allow classes to have a virtual destructor. The following code raises a compile error like "error: attempt to use a deleted function".
//
//     class A {
//     public:
//         virtual ~A();
//         void operator delete(void*) = delete;
//         void operator delete[](void*) = delete;
//     };
//
#define WTF_FORBID_HEAP_ALLOCATION \
private: \
    void* operator new(size_t, void*) = delete; \
    void* operator new[](size_t, void*) = delete; \
    void* operator new(size_t) = delete; \
    void* operator new[](size_t size) = delete; \
    void* operator new(size_t, NotNullTag, void* location) = delete; \
    typedef int __thisIsHereToForceASemicolonAfterThisForbidHeapAllocationMacro
