/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

/*
 A system of macros to make it more ergonomic to implement C++ class member functions in Swift.

 Use HAS_SWIFTCXX_THUNK (from WebGPUExt.h) in headers, to mark a C++ member function as having
 a specialized definition in Swift.

 Use DEFINE_SWIFTCXX_THUNK in implementation sources, to define a function that calls through to a
 "<Class>_<Member>_thunk" free function. These functions are expected to be implemented in Swift and
 available through reverse interop.
 */

#define _DEFINE_SWIFTCXX_THUNK0(Class, Member, ReturnType) \
ReturnType Class::Member() { \
    return Class ## _ ## Member ## _thunk(this); \
}

#define _DEFINE_SWIFTCXX_THUNK1(Class, Member, ReturnType, TypeOfArg1) \
ReturnType Class::Member(TypeOfArg1 arg1) { \
    return Class ## _ ## Member ## _thunk(this, arg1); \
}

#define _DEFINE_SWIFTCXX_THUNK2(Class, Member, ReturnType, TypeOfArg1, TypeOfArg2) \
ReturnType Class::Member(TypeOfArg1 arg1, TypeOfArg2 arg2) { \
    return Class ## _ ## Member ## _thunk(this, arg1, arg2); \
}

#define _DEFINE_SWIFTCXX_THUNK3(Class, Member, ReturnType, TypeOfArg1, TypeOfArg2, TypeOfArg3) \
ReturnType Class::Member(TypeOfArg1 arg1, TypeOfArg2 arg2, TypeOfArg3 arg3) { \
    return Class ## _ ## Member ## _thunk(this, arg1, arg2, arg3); \
}

#define _DEFINE_SWIFTCXX_THUNK4(Class, Member, ReturnType, TypeOfArg1, TypeOfArg2, TypeOfArg3, TypeOfArg4) \
ReturnType Class::Member(TypeOfArg1 arg1, TypeOfArg2 arg2, TypeOfArg3 arg3, TypeOfArg4 arg4) { \
    return Class ## _ ## Member ## _thunk(this, arg1, arg2, arg3, arg4); \
}

#define _DEFINE_SWIFTCXX_THUNK5(Class, Member, ReturnType, TypeOfArg1, TypeOfArg2, TypeOfArg3, TypeOfArg4, TypeOfArg5) \
ReturnType Class::Member(TypeOfArg1 arg1, TypeOfArg2 arg2, TypeOfArg3 arg3, TypeOfArg4 arg4, TypeOfArg5 arg5) { \
    return Class ## _ ## Member ## _thunk(this, arg1, arg2, arg3, arg4, arg5); \
}

#define _GET_NTH_ARG(_1, _2, _3, _4, _5, NAME, ...) NAME

#define DEFINE_SWIFTCXX_THUNK(Class, Member, ReturnType, ...) \
    _GET_NTH_ARG(__VA_ARGS__, _DEFINE_SWIFTCXX_THUNK5, _DEFINE_SWIFTCXX_THUNK4, _DEFINE_SWIFTCXX_THUNK3, _DEFINE_SWIFTCXX_THUNK2, _DEFINE_SWIFTCXX_THUNK1, _DEFINE_SWIFTCXX_THUNK0)(Class, Member, ReturnType, ##__VA_ARGS__)
